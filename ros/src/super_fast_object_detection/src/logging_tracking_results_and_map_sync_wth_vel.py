#!/usr/bin/env python
import rospy, math, random
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
import ros_numpy
import copy
import tf
import time

import message_filters
from sensor_msgs.msg import Image, CompressedImage
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from autoware_msgs.msg import DetectedObject, DetectedObjectArray
from detection_msgs.msg import TrackingObject, TrackingObjectArray
import pandas as pd
from eurecar_lcm_to_ros_publisher.msg import eurecar_can_t
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

class MakeBevImages():
    def __init__(self):
        self.do_overwrite = True

        self.is_callback = False
        
        self.local_map_msg = None
        self.ego_pose_msg = None
        self.velo_msg = None

        self.front_img_msg = None
        self.fl_img_msg = None
        self.fr_img_msg = None

        self.can_msg = None
        self.trk_bboxes_msg = None
        # self.cols_name = ["frame_id", "object_id", "object_type", "position_x", "position_y", "position_z", "object_length", "object_width", "object_height", "heading", "ego_vel", "ego_steer_ang", "ego_x", "ego_y", "ego_heading"]
        
        # object_id: 0 => ego state
        self.cols_name = ["frame_id", "object_id", "object_type", "position_x", "position_y", "position_z", "object_length", "object_width", "object_height", "heading", "ego_vel", "ego_steer_ang", "ego_x", "ego_y", "ego_heading", "other_vx", "other_vy"]

        # Image for time sync
        self.bridge = CvBridge()

        # Message filter
        self.now_global_ego_pose_sub = message_filters.Subscriber("/ego_pose", PoseStamped)
        self.now_local_vector_map_sub = message_filters.Subscriber("/rasterized_map", Image)
        self.now_tracking_bboxes_sub = message_filters.Subscriber("/tracking/objects", TrackingObjectArray)
        self.now_vehicle_status_sub = message_filters.Subscriber("/vehicle/can", eurecar_can_t)
        ts = message_filters.ApproximateTimeSynchronizer([self.now_global_ego_pose_sub, self.now_local_vector_map_sub, self.now_tracking_bboxes_sub, self.now_vehicle_status_sub], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.syncCallback)

        self.publisher = rospy.Publisher('/test', PoseStamped, queue_size=1)

        self.save_dir_path = '/home/usrg/bagfiles/veloster/test'
        self.save_ego_pose_path = os.path.join(self.save_dir_path, 'ego_pose')
        self.save_map_path = os.path.join(self.save_dir_path, 'map')
        if not os.path.exists(self.save_ego_pose_path):
            os.makedirs(self.save_ego_pose_path)
        if not os.path.exists(self.save_map_path):
            os.makedirs(self.save_map_path)

        ego_file_list = os.listdir(self.save_ego_pose_path)
        ego_file_list_txt = [file for file in ego_file_list if file.endswith(".txt")]
        
        if self.do_overwrite:
            self.start_index = 0
        else:
            self.start_index = len(ego_file_list_txt)
        print('start_index', self.start_index)
    
    def syncCallback(self, ego_pose_msg, local_map_msg, trk_bboxes_msg, can_msg):
        self.is_callback = True
        self.ego_pose_msg = ego_pose_msg
        self.local_map_msg = local_map_msg
        self.trk_bboxes_msg = trk_bboxes_msg
        self.can_msg = can_msg

    def save_dataset(self, ego_pose_msg, local_map_msg, trk_bboxes_msg, can_msg):
        if (ego_pose_msg is None or not self.is_callback):
            return
        start_time = time.time()
        
        ####### ego pose #######
        f = open(os.path.join(self.save_ego_pose_path, '%06d.txt'%self.start_index), 'w')
        quaternion = (
                ego_pose_msg.pose.orientation.x,
                ego_pose_msg.pose.orientation.y,
                ego_pose_msg.pose.orientation.z,
                ego_pose_msg.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        ego_yaw = euler[2]
        data = "%f %f %f %f %f %f\n"%(trk_bboxes_msg.header.stamp.to_sec(), trk_bboxes_msg.header.stamp.to_nsec(), ego_pose_msg.pose.position.x, ego_pose_msg.pose.position.y, ego_pose_msg.pose.position.z, ego_yaw)
        f.write(data)
        f.close()

        ####### raster map image #######
        try:
            rasterized_map = self.bridge.imgmsg_to_cv2(local_map_msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        cv2.imwrite(os.path.join(self.save_map_path, '%06d.png'%self.start_index), rasterized_map)
        raster_time = time.time()

        ####### vehicle CAN #######
        ego_vel = can_msg.VS_CAN # km/h
        ego_steer_ang = can_msg.mdps_str_ang # angle

        ####### ego state append ######
        trk_features = []
        feature = [self.start_index, 0, 1, 0., 0., 0., 3.9, 1.6, 1.56, 0, ego_vel, ego_steer_ang, ego_pose_msg.pose.position.x, ego_pose_msg.pose.position.y, ego_yaw, 0, 0]
        trk_features.append(feature)

        ####### track bboxes #######
        for trk in trk_bboxes_msg.objects:
            position = trk.pose.position
            dimensions = trk.dimensions
            quaternion = (
                trk.pose.orientation.x,
                trk.pose.orientation.y,
                trk.pose.orientation.z,
                trk.pose.orientation.w)
            velocity = trk.velocity
            euler = tf.transformations.euler_from_quaternion(quaternion)
            yaw = euler[2]
            feature = [self.start_index, trk.id, int(trk.label), position.x, position.y, position.z, dimensions.x, dimensions.y, dimensions.z, yaw, ego_vel, ego_steer_ang, ego_pose_msg.pose.position.x, ego_pose_msg.pose.position.y, ego_yaw, velocity.linear.x, velocity.linear.y]
            trk_features.append(feature)
        # frame_id, object_id, object_type, position_x, position_y, position_z, object_length, object_width, object_height, heading, ego_vel, ego_steer_ang
        trk_features = np.array(trk_features)
        if len(trk_features.shape) == 1:
            trk_features = np.expand_dims(trk_features, axis=0)

        df = pd.DataFrame(trk_features, columns=self.cols_name)
        csv_file = os.path.join(self.save_dir_path, "agents_status.csv")
        if os.path.isfile(csv_file):  
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file, header=True, index=False)

        print('%06d.npy'%self.start_index)
        self.start_index += 1
        # self.is_callback = False
        end_time = time.time()
        print("%f sec"%(end_time-start_time))
        
        test = PoseStamped()
        test.header = trk_bboxes_msg.header
        self.publisher.publish(test)

    def front_img_callback(self, msg):
        msg.header.stamp = rospy.Time.now()
        self.now_front_img_pub.publish(msg)
    def fl_img_callback(self, msg):
        msg.header.stamp = rospy.Time.now()
        self.now_fl_img_pub.publish(msg)
    def fr_img_callback(self, msg):
        msg.header.stamp = rospy.Time.now()
        self.now_fr_img_pub.publish(msg)

if __name__=='__main__':
    rospy.init_node('logging_lidar_and_map', anonymous=True)
    make_bev = MakeBevImages()
    r = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        make_bev.save_dataset(make_bev.ego_pose_msg, make_bev.local_map_msg, make_bev.trk_bboxes_msg, make_bev.can_msg)
        r.sleep()
    rospy.spin()