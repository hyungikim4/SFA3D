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

class MakeBevImages():
    def __init__(self):
        self.save_RGB_binray_all = True
        self.save_RGB_bev_input = False
        self.do_overwrite = False
        self.with_map = True

        self.is_callback = False
        self.local_map_msg = None
        self.ego_pose_msg = None
        self.velo_msg = None
        self.filtered_velo_msg = None
        self.front_img_msg = None
        self.fl_img_msg = None
        self.fr_img_msg = None

        # Image
        self.front_img_sub = rospy.Subscriber("/gmsl_camera1/compressed", CompressedImage, self.front_img_callback, queue_size=1)
        self.now_front_img_pub = rospy.Publisher("/now/gmsl_camera1/compressed", CompressedImage, queue_size=1)
        self.fl_img_sub = rospy.Subscriber("/gmsl_camera2/compressed", CompressedImage, self.fl_img_callback, queue_size=1)
        self.now_fl_img_pub = rospy.Publisher("/now/gmsl_camera2/compressed", CompressedImage, queue_size=1)
        self.fr_img_sub = rospy.Subscriber("/gmsl_camera3/compressed", CompressedImage, self.fr_img_callback, queue_size=1)
        self.now_fr_img_pub = rospy.Publisher("/now/gmsl_camera3/compressed", CompressedImage, queue_size=1)
        
        # Local vector map
        self.local_vector_map_sub =rospy.Subscriber("/lanes/local_vector_map", PoseArray, self.local_map_callback, queue_size=1)
        self.now_local_vector_map_pub = rospy.Publisher("/now/lanes/local_vector_map", PoseArray, queue_size=1)

        # Ego pose
        self.global_ego_pose_sub = rospy.Subscriber("/ego_pose", PoseStamped, self.ego_pose_callback, queue_size=1)
        self.now_global_ego_pose_pub = rospy.Publisher("/now/ego_pose", PoseStamped, queue_size=1)

        # Lidar
        self.velo_sub = rospy.Subscriber("/transformed_pointcloud", PointCloud2, self.velo_callback, queue_size=1)
        self.now_velo_pub = rospy.Publisher("/now/transformed_pointcloud", PointCloud2, queue_size=1)
        self.filtered_velo_sub = rospy.Subscriber("/points_no_ground", PointCloud2, self.filtered_lidar_callback, queue_size=1)
        self.now_filtered_velo_pub = rospy.Publisher("/now/points_no_ground", PointCloud2, queue_size=1)

        # Message filter
        self.now_global_ego_pose_sub = message_filters.Subscriber("/now/ego_pose", PoseStamped)
        self.now_local_vector_map_sub = message_filters.Subscriber("/now/lanes/local_vector_map", PoseArray)
        self.now_velo_sub = message_filters.Subscriber("/now/transformed_pointcloud", PointCloud2)
        self.now_filtered_velo_sub = message_filters.Subscriber("/now/points_no_ground", PointCloud2)
        self.now_front_img_sub = message_filters.Subscriber("/now/gmsl_camera1/compressed", CompressedImage)
        self.now_fl_img_sub = message_filters.Subscriber("/now/gmsl_camera2/compressed", CompressedImage)
        self.now_fr_img_sub = message_filters.Subscriber("/now/gmsl_camera3/compressed", CompressedImage)
        ts = message_filters.ApproximateTimeSynchronizer([self.now_global_ego_pose_sub, self.now_local_vector_map_sub, self.now_velo_sub, self.now_filtered_velo_sub, self.now_front_img_sub, self.now_fl_img_sub, self.now_fr_img_sub], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.syncCallback)

        self.save_dir_path = '/home/khg/Python_proj/SFA3D/dataset/veloster/training'
        self.save_lidar_path = os.path.join(self.save_dir_path, 'lidar')
        self.save_ego_pose_path = os.path.join(self.save_dir_path, 'ego_pose')
        self.save_front_img_path = os.path.join(self.save_dir_path, 'front_image')
        self.save_front_bev_path = os.path.join(self.save_dir_path, 'front_bev')
        self.save_back_bev_path = os.path.join(self.save_dir_path, 'back_bev')
        self.save_front_label_path = os.path.join(self.save_dir_path, 'front_label')
        self.save_back_label_path = os.path.join(self.save_dir_path, 'back_label')
        if not os.path.exists(self.save_lidar_path):
            os.makedirs(self.save_lidar_path)
        if not os.path.exists(self.save_ego_pose_path):
            os.makedirs(self.save_ego_pose_path)
        if not os.path.exists(self.save_front_img_path):
            os.makedirs(self.save_front_img_path)
        if not os.path.exists(self.save_front_bev_path):
            os.makedirs(self.save_front_bev_path)
        if not os.path.exists(self.save_back_bev_path):
            os.makedirs(self.save_back_bev_path)
        if not os.path.exists(self.save_front_label_path):
            os.makedirs(self.save_front_label_path)
        if not os.path.exists(self.save_back_label_path):
            os.makedirs(self.save_back_label_path)
        lidar_file_list = os.listdir(self.save_lidar_path)
        lidar_file_list_npy = [file for file in lidar_file_list if file.endswith(".npy")]
        
        if self.do_overwrite:
            self.start_index = 0
        else:
            self.start_index = len(lidar_file_list_npy)
        print('start_index', self.start_index)
    
    def syncCallback(self, ego_pose_msg, local_map_msg, velo_msg, filtered_velo_msg,  front_img_msg, fl_img_msg, fr_img_msg):
        self.is_callback = True
        self.ego_pose_msg = ego_pose_msg
        self.local_map_msg = local_map_msg
        self.velo_msg = velo_msg
        self.filtered_velo_msg = filtered_velo_msg
        self.front_img_msg = front_img_msg
        self.fl_img_msg = fl_img_msg
        self.fr_img_msg = fr_img_msg
    
    def get_pixel_xy(self, real_x, real_y):
        x = int(-real_y/cnf.DISCRETIZATION)+cnf.BEV_WIDTH/2
        y = int(-real_x/cnf.DISCRETIZATION)+cnf.BEV_HEIGHT
        return [x,y]

    def drawRasterizedMap(self, map_msg):
        #### Make rasterized map ####
        rasterized_map = np.zeros((2*cnf.BEV_HEIGHT, cnf.BEV_WIDTH, 3), dtype=np.uint8)

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

    def save_dataset(self, ego_pose_msg, local_map_msg, velo_msg, filtered_velo_msg, front_img_msg, fl_img_msg, fr_img_msg):
        if (self.velo_msg is None or not self.is_callback):
            return
        
        f = open(os.path.join(self.save_ego_pose_path, '%06d.txt'%self.start_index), 'w')
        quaternion = (
                ego_pose_msg.pose.orientation.x,
                ego_pose_msg.pose.orientation.y,
                ego_pose_msg.pose.orientation.z,
                ego_pose_msg.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]
        data = "%f %f %f %f\n"%(ego_pose_msg.pose.position.x, ego_pose_msg.pose.position.y, ego_pose_msg.pose.position.z, yaw)
        f.write(data)
        f.close()
        rasterized_map = self.drawRasterizedMap(local_map_msg)
        try:
            np_arr = np.fromstring(front_img_msg.data, np.uint8)
            front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # save image data
        except CvBridgeError as e:
            print(e)
            
        try:
            np_arr = np.fromstring(fl_img_msg.data, np.uint8)
            fl_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # save image data
        except CvBridgeError as e:
            print(e)

        try:
            np_arr = np.fromstring(fr_img_msg.data, np.uint8)
            fr_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # save image data
        except CvBridgeError as e:
            print(e)

        h, w, c = front_img.shape
        clip_w =  int(2.* float(w)/3.)
        front_image_all = np.zeros((h, w + 2*clip_w, c), dtype=np.uint8)
        front_image_all[:, :clip_w, :] = fl_img[:,:clip_w,:]
        front_image_all[:,clip_w:clip_w+w, :] = front_img
        front_image_all[:,clip_w+w:, :] = fr_img[:,w-clip_w:,:]

        # front_image_all = cv2.resize(front_image_all, (int(3*w/2), int(h/2)))
        
        cv2.imwrite(os.path.join(self.save_front_img_path, '%06d.png'%self.start_index), front_image_all)
        
        # save lidar raw data
        msg_numpy = ros_numpy.numpify(velo_msg)
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
        msg_numpy = ros_numpy.numpify(filtered_velo_msg)
        if (len(msg_numpy) == 0):
            return

        if (len(msg_numpy[0]) == 4): # if intensity field exists
            filtered_gen_numpy = get_xyzi_points(msg_numpy, remove_nans=True)
        else:
            xyz_array = ros_numpy.point_cloud2.get_xyz_points(msg_numpy, remove_nans=True)
            i = np.zeros((xyz_array.shape[0], 1))
            filtered_gen_numpy = np.concatenate((xyz_array,i), axis=1)
        # save bev images
        front_lidar = get_filtered_lidar(gen_numpy, cnf.boundary)
        back_lidar = get_filtered_lidar(gen_numpy, cnf.boundary_back)
        filtered_front_lidar = get_filtered_lidar(filtered_gen_numpy, cnf.boundary)
        filtered_back_lidar = get_filtered_lidar(filtered_gen_numpy, cnf.boundary_back)
        front_map = rasterized_map[:cnf.BEV_HEIGHT, :, :]
        back_map = rasterized_map[cnf.BEV_HEIGHT:, :, :]
        if self.save_RGB_binray_all:
            bev_map = makeBEVMap(front_lidar, cnf.boundary)
            bev_map_binary = makeBEVMap_binary(filtered_front_lidar, cnf.boundary)
            back_bevmap = makeBEVMap(back_lidar, cnf.boundary_back)
            back_bevmap_binary = makeBEVMap_binary(filtered_back_lidar, cnf.boundary_back)

            bev_map = np.transpose(bev_map, (1,2,0))
            back_bevmap = np.transpose(back_bevmap, (1,2,0))
            bev_map_binary = np.transpose(bev_map_binary, (1,2,0))
            back_bevmap_binary = np.transpose(back_bevmap_binary, (1,2,0))
        
            all_bev_map = np.zeros((cnf.BEV_HEIGHT, 2*cnf.BEV_WIDTH, 3))
            all_back_bev_map = np.zeros((cnf.BEV_HEIGHT, 2*cnf.BEV_WIDTH, 3))

            all_bev_map[:,:cnf.BEV_WIDTH,:] = bev_map
            all_bev_map[:,cnf.BEV_WIDTH:,:] = bev_map_binary

            all_back_bev_map[:,:cnf.BEV_WIDTH,:] = back_bevmap
            all_back_bev_map[:,cnf.BEV_WIDTH:,:] = back_bevmap_binary
            
            all_bev_map = (all_bev_map*255).astype(np.uint8)
            all_back_bev_map = (all_back_bev_map*255).astype(np.uint8)

            all_bev_map = cv2.rotate(all_bev_map, cv2.ROTATE_180)
            all_back_bev_map = cv2.rotate(all_back_bev_map, cv2.ROTATE_180)

            bev_map = all_bev_map
            back_bevmap = all_back_bev_map
            if (self.with_map):
                bev_map[:,:cnf.BEV_WIDTH,:] = cv2.addWeighted(bev_map[:,:cnf.BEV_WIDTH,:],0.8,front_map,0.2,0)
                back_bevmap[:,:cnf.BEV_WIDTH,:] = cv2.addWeighted(back_bevmap[:,:cnf.BEV_WIDTH,:],0.8,back_map,0.2,0)
            # output_bev_map = np.zeros((2*cnf.BEV_HEIGHT, 2*cnf.BEV_WIDTH, 3))
            # output_bev_map[:cnf.BEV_HEIGHT,:,:] = all_bev_map
            # output_bev_map[cnf.BEV_HEIGHT:,:,:] = cv2.resize(front_img, (2*cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            
            # bev_map = output_bev_map
            # back_bevmap = all_back_bev_map 
        else:
            if self.save_RGB_bev_input:
                bev_map = makeBEVMap(front_lidar, cnf.boundary)
                back_bevmap = makeBEVMap(back_lidar, cnf.boundary_back)
            else:
                bev_map = makeBEVMap_binary(front_lidar, cnf.boundary)
                back_bevmap = makeBEVMap_binary(back_lidar, cnf.boundary_back)
        
            bev_map = np.transpose(bev_map, (1,2,0))
            back_bevmap = np.transpose(back_bevmap, (1,2,0))
            bev_map = (bev_map*255).astype(np.uint8)
            back_bevmap = (back_bevmap*255).astype(np.uint8)
            bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
            back_bevmap = cv2.rotate(back_bevmap, cv2.ROTATE_180)

        # cv2.imshow('bev', bev_map)
        # cv2.imshow('back', back_bevmap)
        # cv2.waitKey(1)

        cv2.imwrite(os.path.join(self.save_front_bev_path, '%06d.png'%self.start_index), bev_map)
        cv2.imwrite(os.path.join(self.save_back_bev_path, '%06d.png'%self.start_index), back_bevmap)
        print('%06d.png'%self.start_index)
        self.start_index += 1
        self.is_callback = False

    def local_map_callback(self, msg):
        msg.header.stamp = rospy.Time.now()
        self.now_local_vector_map_pub.publish(msg)
    def ego_pose_callback(self, msg):
        msg.header.stamp = rospy.Time.now()
        self.now_global_ego_pose_pub.publish(msg)
    def front_img_callback(self, msg):
        msg.header.stamp = rospy.Time.now()
        self.now_front_img_pub.publish(msg)
    def fl_img_callback(self, msg):
        msg.header.stamp = rospy.Time.now()
        self.now_fl_img_pub.publish(msg)
    def fr_img_callback(self, msg):
        msg.header.stamp = rospy.Time.now()
        self.now_fr_img_pub.publish(msg)
    def velo_callback(self, msg):
        msg.header.stamp = rospy.Time.now()
        self.now_velo_pub.publish(msg)
    def filtered_lidar_callback(self, msg):
        msg.header.stamp = rospy.Time.now()
        self.now_filtered_velo_pub.publish(msg)
if __name__=='__main__':
    rospy.init_node('make_bin_and_bev_images', anonymous=True)
    make_bev = MakeBevImages()
    r = rospy.Rate(1) # 10hz
    while not rospy.is_shutdown():
        make_bev.save_dataset(make_bev.ego_pose_msg, make_bev.local_map_msg, make_bev.velo_msg, make_bev.filtered_velo_msg, make_bev.front_img_msg, make_bev.fl_img_msg, make_bev.fr_img_msg)
        r.sleep()
    rospy.spin()