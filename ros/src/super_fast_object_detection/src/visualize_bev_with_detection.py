#!/usr/bin/env python
import rospy, math, random
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from laser_geometry import LaserProjection
import sensor_msgs.point_cloud2 as pc2
import ros_numpy
import copy
import time
import tf

import message_filters
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
#import matplotlib.pyplot as plt
import os.path
import os
import glob
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
# sys.path.append('/home/khg/Python_proj/SFA3D')
from veloster_bev_utils import makeBEVMap, makeBEVMap_binary
import veloster_config as cnf
from veloster_data_utils import get_filtered_lidar


def get_xyzi_points(cloud_array, remove_nans=True, dtype=np.float):
    '''Pulls out x, y, and z columns from the cloud recordarray, and returns
	a 3xN matrix.
    '''
    # remove crap points
    # print(cloud_array.dtype.names)
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z']) & np.isfinite(cloud_array['intensity'])
        cloud_array = cloud_array[mask]
    
    # pull out x, y, and z values + intensity
    points = np.zeros(cloud_array.shape + (4,), dtype=dtype)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']
    points[...,3] = cloud_array['intensity']

    return points

class VisualizeBevWithDetection():
    def __init__(self):
        self.bridge = CvBridge()
        self.is_callback = False
        self.with_map = False
        self.white_background = False
        self.velo_msg = None
        self.detection_bboxes_msg = None
        self.local_map_msg = None

        # Message filter
        # self.velo_sub = message_filters.Subscriber("/transformed_pointcloud", PointCloud2)
        # self.detection_bboxes_sub = message_filters.Subscriber("/detection/bboxes", BoundingBoxArray)
        # self.local_vector_map_sub =message_filters.Subscriber("/lanes/local_vector_map", PoseArray)
        # ts = message_filters.ApproximateTimeSynchronizer([self.velo_sub, self.detection_bboxes_sub, self.local_vector_map_sub], 10, 0.1, allow_headerless=True)
        # ts.registerCallback(self.syncCallback)

        self.velo_sub = rospy.Subscriber("/transformed_pointcloud", PointCloud2, self.velo_callback, queue_size=1)
        self.detection_bboxes_sub = rospy.Subscriber("/detection/bboxes", BoundingBoxArray, self.detection_bboxes_callback, queue_size=1)
        self.local_vector_map_sub = rospy.Subscriber("/lanes/local_vector_map", PoseArray, self.local_map_callback, queue_size=1)

        # publish visualization results
        self.visual_result_pub =rospy.Publisher("/visualization/detection", Image, queue_size=1)
    
    def syncCallback(self, velo_msg, detection_bboxes_msg, local_map_msg):
        self.is_callback = True
        self.velo_msg = velo_msg
        self.detection_bboxes_msg = detection_bboxes_msg
        self.local_map_msg = local_map_msg
    

    def velo_callback(self, velo_msg):
        self.is_callback = True
        self.velo_msg = velo_msg
    def detection_bboxes_callback(self, detection_msg):
        self.detection_bboxes_msg = detection_msg
    def local_map_callback(self, map_msg):
        self.local_map_msg = map_msg

    def rot_z(self, yaw):
        return np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)]])

    def get_pixel_xy(self, real_x, real_y):
        x = int(-real_y/cnf.DISCRETIZATION)+cnf.BEV_WIDTH/2
        y = int(-real_x/cnf.DISCRETIZATION)+cnf.BEV_HEIGHT
        return [x,y]
    
    def get_bev_bboxes(self, msg):
        bev_clusters_bboxes = []
        for bbox in msg.boxes:
            xy = np.array([[bbox.pose.position.x],[bbox.pose.position.y]])
            dx = bbox.dimensions.x
            dy = bbox.dimensions.y
            corners_bev = np.array([[dx/2., dx/2., -dx/2., -dx/2.],
                                    [dy/2., -dy/2., dy/2., -dy/2.]])
            quaternion = (
                bbox.pose.orientation.x,
                bbox.pose.orientation.y,
                bbox.pose.orientation.z,
                bbox.pose.orientation.w)
            roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)

            rot_matrix = self.rot_z(yaw)
            rotated_corners_bev = np.dot(rot_matrix, corners_bev)
            corners_bev = rotated_corners_bev + xy
            bev_clusters_bboxes.append(corners_bev.transpose())
        return np.array(bev_clusters_bboxes)

    def drawObjects(self, bev_bboxes, image):
        for bev_bbox in bev_bboxes:    
            corner_2d = []
            for i in range(len(bev_bbox)):
                corner_2d.append(self.get_pixel_xy(bev_bbox[i][0], bev_bbox[i][1]))
            corner_2d = np.array(corner_2d)
            # print(corner_2d)
            corners_2d_dim = np.expand_dims(corner_2d, axis=1)
            corners_2d_dim = corners_2d_dim.astype(int)
            hull = cv2.convexHull(corners_2d_dim)
            
            cv2.drawContours(image, [hull], 0, (0, 255, 0), thickness=1)
        return image


    def drawRasterizedMap(self, map_msg):
        #### Make rasterized map ####
        rasterized_map = np.zeros((2*cnf.BEV_HEIGHT, cnf.BEV_WIDTH, 3), dtype=np.uint8)

        if (map_msg is None):
            return rasterized_map
        # Get each lane
        num_of_pose = int(len(map_msg.poses)/2)
        left_pose_array = map_msg.poses[:num_of_pose]
        right_pose_array = map_msg.poses[num_of_pose:]    
        
        left_lanes = []
        right_lanes = []
        left_lane_pts = []
        right_lane_pts = []
        for i in range(num_of_pose - 1):
            now_left_pose = left_pose_array[i]
            next_left_pose = left_pose_array[i+1]
            if (now_left_pose.position.z != next_left_pose.position.z):
                left_lanes.append(np.array(left_lane_pts))
                right_lanes.append(np.array(right_lane_pts))
                left_lane_pts = []
                right_lane_pts = []
                continue 

            now_right_pose = right_pose_array[i]
            
            left_lane_pts.append(self.get_pixel_xy(now_left_pose.position.x, now_left_pose.position.y))
            right_lane_pts.append(self.get_pixel_xy(now_right_pose.position.x, now_right_pose.position.y))

        
        if (len(left_pose_array) != 0):
            now_left_pose = left_pose_array[len(left_pose_array)-1]
            now_right_pose = right_pose_array[len(right_pose_array)-1]
            left_lane_pts.append(self.get_pixel_xy(now_left_pose.position.x, now_left_pose.position.y))
            right_lane_pts.append(self.get_pixel_xy(now_right_pose.position.x, now_right_pose.position.y))

            left_lanes.append(np.array(left_lane_pts))
            right_lanes.append(np.array(right_lane_pts))
            left_lane_pts = []
            right_lane_pts = []
        
        # draw lanes and segments (drivable area)
        for i in range(len(left_lanes)):
            left_lane_pts = left_lanes[i]
            if (left_lane_pts.shape[0] == 0):
                continue
            right_lane_pts = right_lanes[i]
            # draw segments (drivable area)
            side_lane_pts = np.concatenate((left_lane_pts, np.flip(right_lane_pts, 0)), axis=0)
            cv2.fillPoly(rasterized_map, pts =[side_lane_pts], color=(128,128,128)) # (128,128,128)

        for i in range(len(left_lanes)):
            left_lane_pts = left_lanes[i]
            if (left_lane_pts.shape[0] == 0):
                continue
            right_lane_pts = right_lanes[i]
            # draw lanes
            for j in range(len(left_lane_pts) - 1):
                cv2.line(rasterized_map, (left_lane_pts[j][0], left_lane_pts[j][1]), (left_lane_pts[j+1][0], left_lane_pts[j+1][1]),(0,255,0), 2) # (0, 255, 0), 2
                cv2.line(rasterized_map, (right_lane_pts[j][0], right_lane_pts[j][1]), (right_lane_pts[j+1][0], right_lane_pts[j+1][1]),(0,255,0), 2)
        return rasterized_map

    def makeRawLidarBevWithDetection(self, velo_msg, local_map_msg, detection_msg):
        if (velo_msg is None or not self.is_callback):
            return

        # make raw lidar bev
        msg_numpy = ros_numpy.numpify(velo_msg)
        if (len(msg_numpy) == 0):
            return

        if (len(msg_numpy[0]) == 4): # if intensity field exists
            gen_numpy = get_xyzi_points(msg_numpy, remove_nans=True)
        else:
            xyz_array = ros_numpy.point_cloud2.get_xyz_points(msg_numpy, remove_nans=True)
            i = np.zeros((xyz_array.shape[0], 1))
            gen_numpy = np.concatenate((xyz_array,i), axis=1)
        
        # Draw raw lidar bev image
        front_lidar = get_filtered_lidar(gen_numpy, cnf.boundary)
        back_lidar = get_filtered_lidar(gen_numpy, cnf.boundary_back)

        bev_map_binary = makeBEVMap_binary(front_lidar, cnf.boundary)
        back_bevmap_binary = makeBEVMap_binary(back_lidar, cnf.boundary_back)

        bev_map_binary = np.transpose((bev_map_binary*255).astype(np.uint8), (1,2,0))
        back_bevmap_binary = np.transpose((back_bevmap_binary*255).astype(np.uint8), (1,2,0))

        bev_map_binary = cv2.rotate(bev_map_binary, cv2.ROTATE_180)
        back_bevmap_binary = cv2.rotate(back_bevmap_binary, cv2.ROTATE_180)

        if self.white_background:
            bev_map_binary = 255 - bev_map_binary
            back_bevmap_binary = 255 - back_bevmap_binary
        # Draw objects in front image
        if (detection_msg is not None):
            bev_bboxes = self.get_bev_bboxes(detection_msg)
            bev_map_binary = self.drawObjects(bev_bboxes, bev_map_binary)
        
        # Draw raw lidar bev with raster map
        if (self.with_map):
            rasterized_map = self.drawRasterizedMap(local_map_msg)
            front_map = rasterized_map[:cnf.BEV_HEIGHT, :, :]
            back_map = rasterized_map[cnf.BEV_HEIGHT:, :, :]
            bev_map_binary = cv2.addWeighted(bev_map_binary,0.8,front_map,0.2,0)
            back_bevmap_binary = cv2.addWeighted(back_bevmap_binary,0.8,back_map,0.2,0)

        all_bev_map = np.zeros((2*cnf.BEV_HEIGHT, cnf.BEV_WIDTH, 3), dtype=np.uint8)
        all_bev_map[:cnf.BEV_HEIGHT, :, :] = bev_map_binary
        all_bev_map[cnf.BEV_HEIGHT:, :, :] = back_bevmap_binary

        # publish visualization results
        all_bev_map_msg = self.bridge.cv2_to_imgmsg(all_bev_map, "bgr8")
        all_bev_map_msg.header = velo_msg.header
        self.visual_result_pub.publish(all_bev_map_msg)

        self.is_callback = False

if __name__=='__main__':
    rospy.init_node('visualize_bev_with_detection', anonymous=True)
    visualize_bev = VisualizeBevWithDetection()
    while not rospy.is_shutdown():
        visualize_bev.makeRawLidarBevWithDetection(visualize_bev.velo_msg, visualize_bev.local_map_msg, visualize_bev.detection_bboxes_msg)
    rospy.spin()