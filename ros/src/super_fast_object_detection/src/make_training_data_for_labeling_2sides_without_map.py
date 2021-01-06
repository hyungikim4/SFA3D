#!/usr/bin/env python
import rospy, math, random
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from laser_geometry import LaserProjection
import sensor_msgs.point_cloud2 as pc2
import ros_numpy
import copy
import tf
import time

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
from veloster_2sides_bev_utils import makeBEVMap, makeBEVMap_binary
import veloster_config_2sides as cnf
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

class MakeBevImages():
    def __init__(self):
        self.save_RGB_binary_all = True
        self.save_RGB_bev_input = True
        self.do_overwrite = True
        self.with_map = True

        self.is_callback = False
        self.lidar_msg = None
        self.filtered_lidar_msg = None
        self.front_img_msg = None

        # Message filter
        self.lidar_sub = message_filters.Subscriber("/transformed_pointcloud", PointCloud2)
        self.filtered_lidar_sub = message_filters.Subscriber("/points_no_ground", PointCloud2)
        self.front_img_sub = message_filters.Subscriber("/vds_node_localhost_2218/image_raw/compressed", CompressedImage)
        ts = message_filters.ApproximateTimeSynchronizer([self.lidar_sub, self.filtered_lidar_sub, self.front_img_sub], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.syncCallback)

        self.save_dir_path = '/home/khg/Python_proj/SFA3D/dataset/veloster_2sides/training/hyundai_test'
        self.save_lidar_path = os.path.join(self.save_dir_path, 'lidar')
        self.save_front_img_path = os.path.join(self.save_dir_path, 'front_image')
        self.save_bev_path = os.path.join(self.save_dir_path, 'bev')
        self.save_label_path = os.path.join(self.save_dir_path, 'label')
        if not os.path.exists(self.save_lidar_path):
            os.makedirs(self.save_lidar_path)
        if not os.path.exists(self.save_front_img_path):
            os.makedirs(self.save_front_img_path)
        if not os.path.exists(self.save_bev_path):
            os.makedirs(self.save_bev_path)
        if not os.path.exists(self.save_label_path):
            os.makedirs(self.save_label_path)

        lidar_file_list = os.listdir(self.save_lidar_path)
        lidar_file_list_npy = [file for file in lidar_file_list if file.endswith(".npy")]
        
        if self.do_overwrite:
            self.start_index = 0
        else:
            self.start_index = len(lidar_file_list_npy)
        print('start_index', self.start_index)
    
    def syncCallback(self, lidar_msg, filtered_lidar_msg, front_img_msg,):
        self.is_callback = True
        self.lidar_msg = lidar_msg
        self.filtered_lidar_msg = filtered_lidar_msg
        self.front_img_msg = front_img_msg
    
    def get_pixel_xy(self, real_x, real_y):
        x = int(-real_y/cnf.DISCRETIZATION)+cnf.BEV_WIDTH/2
        y = int(-real_x/cnf.DISCRETIZATION)+cnf.BEV_HEIGHT/2
        return [x,y]

    def drawRasterizedMap(self, map_msg):
        #### Make rasterized map ####
        rasterized_map = np.zeros((cnf.BEV_HEIGHT, cnf.BEV_WIDTH, 3), dtype=np.uint8)

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

    def save_dataset(self, lidar_msg, filtered_lidar_msg, front_img_msg):
        if (lidar_msg is None or not self.is_callback):
            return
        start_time = time.time()    

        try:
            np_arr = np.fromstring(front_img_msg.data, np.uint8)
            front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # save image data
        except CvBridgeError as e:
            print(e)
            
        h, w, c = front_img.shape
        cv2.imwrite(os.path.join(self.save_front_img_path, '%06d.png'%self.start_index), front_img)
        img_time = time.time()

        # print("img time %.4f"%(img_time-raster_time))     
        # save lidar raw data
        msg_numpy = ros_numpy.numpify(lidar_msg)
        if (len(msg_numpy) == 0):
            return

        if (len(msg_numpy[0]) == 4): # if intensity field exists
            gen_numpy = get_xyzi_points(msg_numpy, remove_nans=True)
        else:
            xyz_array = ros_numpy.point_cloud2.get_xyz_points(msg_numpy, remove_nans=True)
            i = np.zeros((xyz_array.shape[0], 1))
            gen_numpy = np.concatenate((xyz_array,i), axis=1)
        
        np.save(os.path.join(self.save_lidar_path, '%06d'%self.start_index), gen_numpy)
        
        # Filtered lidar (Ground filter)
        msg_numpy = ros_numpy.numpify(filtered_lidar_msg)
        if (len(msg_numpy) == 0):
            return

        if (len(msg_numpy[0]) == 4): # if intensity field exists
            filtered_gen_numpy = get_xyzi_points(msg_numpy, remove_nans=True)
        else:
            xyz_array = ros_numpy.point_cloud2.get_xyz_points(msg_numpy, remove_nans=True)
            i = np.zeros((xyz_array.shape[0], 1))
            filtered_gen_numpy = np.concatenate((xyz_array,i), axis=1)
        
        # save bev images
        lidar = get_filtered_lidar(gen_numpy, cnf.boundary)
        filtered_lidar = get_filtered_lidar(filtered_gen_numpy, cnf.boundary)

        if self.save_RGB_binary_all:
            bev_map = makeBEVMap(lidar, cnf.boundary)
            bev_map_binary = makeBEVMap_binary(filtered_lidar, cnf.boundary)

            bev_map = np.transpose(bev_map, (1,2,0))
            bev_map_binary = np.transpose(bev_map_binary, (1,2,0))
        
            all_bev_map = np.zeros((cnf.BEV_HEIGHT, 2*cnf.BEV_WIDTH, 3))

            all_bev_map[:,:cnf.BEV_WIDTH,:] = bev_map
            all_bev_map[:,cnf.BEV_WIDTH:,:] = bev_map_binary
            
            all_bev_map = (all_bev_map*255).astype(np.uint8)
            all_bev_map = cv2.rotate(all_bev_map, cv2.ROTATE_180)

            bev_map = all_bev_map
 
        else:
            if self.save_RGB_bev_input:
                bev_map = makeBEVMap(lidar, cnf.boundary)
            else:
                bev_map = makeBEVMap_binary(lidar, cnf.boundary)
        
            bev_map = np.transpose(bev_map, (1,2,0))
            bev_map = (bev_map*255).astype(np.uint8)
            bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

        bev_time = time.time()
        # print("bev time %.4f"%(bev_time-img_time))

        cv2.imwrite(os.path.join(self.save_bev_path, '%06d.png'%self.start_index), bev_map)

        print('%06d.png'%self.start_index)
        self.start_index += 1
        self.is_callback = False
        end_time = time.time()
        # print("%f sec"%(end_time-start_time))

if __name__=='__main__':
    rospy.init_node('make_bin_and_bev_images', anonymous=True)
    make_bev = MakeBevImages()
    r = rospy.Rate(1) # 10hz
    while not rospy.is_shutdown():
        make_bev.save_dataset(make_bev.lidar_msg, make_bev.filtered_lidar_msg, make_bev.front_img_msg)
        r.sleep()
    rospy.spin()