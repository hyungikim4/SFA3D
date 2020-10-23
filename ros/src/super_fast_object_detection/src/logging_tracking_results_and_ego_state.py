#!/usr/bin/env python
import rospy, math, random
import numpy as np
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

class LoggingTrackingResultsAndEgoState():
    def __init__(self):
        self.ego_pose_msg = None
        self.ego_can_msg = None
        self.trk_obj_msg = None

        self.start_index = 0

        # object_id: 0 => ego state
        self.cols_name = ["frame_id", "object_id", "object_type", "position_x", "position_y", "position_z", "object_length", "object_width", "object_height", "heading", "ego_vel", "ego_steer_ang", "ego_x", "ego_y", "ego_heading", "other_vx", "other_vy"]

        # Message filter
        self.vehicle_pose_sub = message_filters.Subscriber("/vehicle/pose", Odometry)
        self.vehicle_can_sub = message_filters.Subscriber("/vehicle/can", eurecar_can_t)
        self.tracking_objects_sub = message_filters.Subscriber("/tracking/objects", TrackingObjectArray)
        ts = message_filters.ApproximateTimeSynchronizer([self.vehicle_pose_sub, self.vehicle_can_sub, self.tracking_objects_sub], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.syncCallback)

        self.publisher = rospy.Publisher('/test', PoseStamped, queue_size=1)

        self.save_dir_path = '/home/usrg/bagfiles/veloster/test'
        if not os.path.exists(self.save_dir_path):
            os.makedirs(self.save_dir_path)

    def syncCallback(self, ego_pose_msg, ego_can_msg, trk_obj_msg):
        self.ego_pose_msg = ego_pose_msg
        self.ego_can_msg = ego_can_msg
        self.trk_obj_msg = trk_obj_msg

    def save_dataset(self, ego_pose_msg, ego_can_msg, trk_obj_msg):
        if (ego_pose_msg is None):
            return
        start_time = time.time()
        
        ####### vehicle CAN #######
        ego_vel = ego_can_msg.VS_CAN # km/h
        ego_steer_ang = ego_can_msg.mdps_str_ang # angle

        ####### ego state append ######
        trk_features = []
        ego_position = ego_pose_msg.pose.pose.position
        quaternion = (
                ego_pose_msg.pose.pose.orientation.x,
                ego_pose_msg.pose.pose.orientation.y,
                ego_pose_msg.pose.pose.orientation.z,
                ego_pose_msg.pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        ego_yaw = euler[2]
        feature = [self.start_index, 0, 1, 0., 0., 0., 3.9, 1.6, 1.56, 0, ego_vel, ego_steer_ang, ego_position.x, ego_position.y, ego_yaw, 0, 0]
        trk_features.append(feature)

        ####### track bboxes #######
        for trk in trk_obj_msg.objects:
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
            feature = [self.start_index, trk.id, int(trk.label), position.x, position.y, position.z, dimensions.x, dimensions.y, dimensions.z, yaw, ego_vel, ego_steer_ang, ego_position.x, ego_position.y, ego_yaw, velocity.linear.x, velocity.linear.y]
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

        print('frame_id: %d'%self.start_index)
        self.start_index += 1
        # self.is_callback = False
        end_time = time.time()
        print("%f sec"%(end_time-start_time))
        
        test = PoseStamped()
        test.header = trk_obj_msg.header
        self.publisher.publish(test)

if __name__=='__main__':
    rospy.init_node('logging_tracking_results_and_ego_state', anonymous=True)
    logging_state = LoggingTrackingResultsAndEgoState()
    r = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        logging_state.save_dataset(logging_state.ego_pose_msg, logging_state.ego_can_msg, logging_state.trk_obj_msg)
        r.sleep()
    rospy.spin()