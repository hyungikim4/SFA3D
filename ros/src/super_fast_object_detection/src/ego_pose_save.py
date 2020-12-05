#!/usr/bin/env python
import rospy, math, random
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import ros_numpy
import copy
import tf
import time

import message_filters
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from detection_msgs.msg import TrackingObject, TrackingObjectArray
import pandas as pd
from eurecar_lcm_to_ros_publisher.msg import eurecar_can_t
import os
import glob
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

class EgoPose():
    def __init__(self):
        base_dir = '/media/khg/5E103DDF103DBF39/veloster_rosbag/veloster_tracking_dataset/2020-09-22-16-08-48'
        ego_pose_dir = os.path.join(base_dir, 'ego_pose')
        ego_pose_filelist = sorted(glob.glob(os.path.join(ego_pose_dir, '*.txt')))
        self.ego_pose_msg = None

        filenames = []
        timestamp_sec = []
        timestamp_nsec = []
        for ego_file in ego_pose_filelist:
            ego_idx = os.path.basename(ego_file)[:-4]
            filenames.append(ego_idx)
            with open(ego_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    str_values = line.split(' ')
                    timestamp_sec.append(float(str_values[0]))
                    timestamp_nsec.append(float(str_values[1]))

        self.filenames =  np.array(filenames)
        self.timestamp_sec = np.array(timestamp_sec)
        self.timestamp_nsec = np.array(timestamp_nsec)

        print(len(self.timestamp_sec), len(self.filenames))

        # Message filter
        self.lidar_sub = message_filters.Subscriber("/transformed_pointcloud", PointCloud2)
        self.vehicle_pose_sub = message_filters.Subscriber("/vehicle/pose", Odometry)

        self.publisher = rospy.Publisher('/test', PoseStamped, queue_size=1)

        ts = message_filters.ApproximateTimeSynchronizer([self.lidar_sub, self.vehicle_pose_sub], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.syncCallback)

        self.save_dir_path = os.path.join(base_dir, 'ego_pose_rere')
        if not os.path.exists(self.save_dir_path):
            os.makedirs(self.save_dir_path)
        
        self.num = 0

    def syncCallback(self, lidar_msg, ego_pose_msg):
        lidar_stamp = lidar_msg.header.stamp
        seconds = lidar_stamp.to_sec()
        nanoseconds = lidar_stamp.to_nsec()

        # index = np.where((self.timestamp_sec == seconds) & (self.timestamp_nsec == nanoseconds))
        index = np.where((self.timestamp_sec == seconds))
        print(seconds, nanoseconds, self.filenames[index])
        self.num += 1
        print(self.num)

        test = PoseStamped()
        test.header = lidar_msg.header
        self.publisher.publish(test)
        
    def vehicle_pose_callback(self, ego_pose_msg):
        self.ego_pose_msg = ego_pose_msg

    def save_ego_pose(self, ego_pose_msg):
        if ego_pose_msg is None:
            return
        ego_stamp = ego_pose_msg.header.stamp
        seconds = ego_stamp.to_sec()
        nanoseconds = ego_stamp.to_nsec()
        
        pose = ego_pose_msg.pose.pose.position
        
        quaternion = (
                ego_pose_msg.pose.pose.orientation.x,
                ego_pose_msg.pose.pose.orientation.y,
                ego_pose_msg.pose.pose.orientation.z,
                ego_pose_msg.pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]

        save_path = os.path.join(self.save_dir_path, "%06d.txt"%self.num)
        with open(save_path, 'w') as f:
            data = "%f %f %f %f %f %f\n"%(seconds, nanoseconds, pose.x, pose.y, pose.z, yaw)
            f.write(data)
        self.num += 1
        
        # index = np.where((self.timestamp_sec == seconds) & (self.timestamp_nsec == nanoseconds))
        # index = np.where((self.timestamp_sec == seconds))
        # print(seconds, nanoseconds, self.filenames[index])
        # self.num += 1
        # print(self.num)
        
        
if __name__=='__main__':
    rospy.init_node('ego_pose_save', anonymous=True)
    ego_pose_save = EgoPose()
    # r = rospy.Rate(30) # 10hz
    # while not rospy.is_shutdown():
    #     ego_pose_save.save_ego_pose(ego_pose_save.ego_pose_msg)
    #     r.sleep()
    rospy.spin()