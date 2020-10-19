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
        self.cols_name = ["frame_id", "object_id", "object_type", "position_x", "position_y", "position_z", "object_length", "object_width", "object_height", "heading", "ego_vel", "ego_steer_ang"]

        # Image for time sync
        self.front_img_sub = rospy.Subscriber("/gmsl_camera1/DwODresult_img/compressed", CompressedImage, self.front_img_callback, queue_size=1)
        self.now_front_img_pub = rospy.Publisher("/now/gmsl_camera1/compressed", CompressedImage, queue_size=1)
        self.fl_img_sub = rospy.Subscriber("/gmsl_camera2/DwODresult_img/compressed", CompressedImage, self.fl_img_callback, queue_size=1)
        self.now_fl_img_pub = rospy.Publisher("/now/gmsl_camera2/compressed", CompressedImage, queue_size=1)
        self.fr_img_sub = rospy.Subscriber("/gmsl_camera3/DwODresult_img/compressed", CompressedImage, self.fr_img_callback, queue_size=1)
        self.now_fr_img_pub = rospy.Publisher("/now/gmsl_camera3/compressed", CompressedImage, queue_size=1)

        # Message filter
        self.now_global_ego_pose_sub = message_filters.Subscriber("/ego_pose", PoseStamped)
        self.now_local_vector_map_sub = message_filters.Subscriber("/rasterized_map", Image)
        self.now_velo_sub = message_filters.Subscriber("/transformed_pointcloud", PointCloud2)
        self.now_front_img_sub = message_filters.Subscriber("/now/gmsl_camera1/compressed", CompressedImage)
        self.now_fl_img_sub = message_filters.Subscriber("/now/gmsl_camera2/compressed", CompressedImage)
        self.now_fr_img_sub = message_filters.Subscriber("/now/gmsl_camera3/compressed", CompressedImage)
        self.now_tracking_bboxes_sub = message_filters.Subscriber("/tracking/bboxes", DetectedObjectArray)
        self.now_vehicle_status_sub = message_filters.Subscriber("/vehicle/can", eurecar_can_t)
        ts = message_filters.ApproximateTimeSynchronizer([self.now_global_ego_pose_sub, self.now_local_vector_map_sub, self.now_velo_sub, self.now_front_img_sub, self.now_fl_img_sub, self.now_fr_img_sub, self.now_tracking_bboxes_sub, self.now_vehicle_status_sub], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.syncCallback)

        self.publisher = rospy.Publisher('/test', PoseStamped, queue_size=1)

        self.save_dir_path = '/media/khg/5E103DDF103DBF39/veloster_rosbag/test'
        self.save_lidar_path = os.path.join(self.save_dir_path, 'lidar')
        self.save_ego_pose_path = os.path.join(self.save_dir_path, 'ego_pose')
        self.save_front_img_path = os.path.join(self.save_dir_path, 'front_image')
        self.save_map_path = os.path.join(self.save_dir_path, 'map')
        if not os.path.exists(self.save_lidar_path):
            os.makedirs(self.save_lidar_path)
        if not os.path.exists(self.save_ego_pose_path):
            os.makedirs(self.save_ego_pose_path)
        if not os.path.exists(self.save_front_img_path):
            os.makedirs(self.save_front_img_path)
        if not os.path.exists(self.save_map_path):
            os.makedirs(self.save_map_path)

        lidar_file_list = os.listdir(self.save_lidar_path)
        lidar_file_list_npy = [file for file in lidar_file_list if file.endswith(".npy")]
        
        if self.do_overwrite:
            self.start_index = 0
        else:
            self.start_index = len(lidar_file_list_npy)
        print('start_index', self.start_index)
    
    def syncCallback(self, ego_pose_msg, local_map_msg, velo_msg, front_img_msg, fl_img_msg, fr_img_msg, trk_bboxes_msg, can_msg):
        print('syncCallback')
        self.is_callback = True
        self.ego_pose_msg = ego_pose_msg
        self.local_map_msg = local_map_msg
        self.velo_msg = velo_msg
        self.front_img_msg = front_img_msg
        self.fl_img_msg = fl_img_msg
        self.fr_img_msg = fr_img_msg
        self.trk_bboxes_msg = trk_bboxes_msg
        self.can_msg = can_msg

    def save_dataset(self, ego_pose_msg, local_map_msg, velo_msg, front_img_msg, fl_img_msg, fr_img_msg, trk_bboxes_msg, can_msg):
        if (velo_msg is None or ego_pose_msg is None or not self.is_callback):
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
        yaw = euler[2]
        data = "%f %f %f %f %f %f\n"%(velo_msg.header.stamp.to_sec(), velo_msg.header.stamp.to_nsec(), ego_pose_msg.pose.position.x, ego_pose_msg.pose.position.y, ego_pose_msg.pose.position.z, yaw)
        f.write(data)
        f.close()

        ####### raster map image #######
        try:
            np_arr = np.fromstring(local_map_msg.data, np.uint8)
            rasterized_map = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except CvBridgeError as e:
            print(e)

        cv2.imwrite(os.path.join(self.save_map_path, '%06d.png'%self.start_index), rasterized_map)
        raster_time = time.time()

        ####### front image #######
        try:
            np_arr = np.fromstring(front_img_msg.data, np.uint8)
            front_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except CvBridgeError as e:
            print(e)
            
        try:
            np_arr = np.fromstring(fl_img_msg.data, np.uint8)
            fl_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except CvBridgeError as e:
            print(e)

        try:
            np_arr = np.fromstring(fr_img_msg.data, np.uint8)
            fr_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except CvBridgeError as e:
            print(e)

        h, w, c = front_img.shape
        clip_w =  int(2.* float(w)/3.)
        front_image_all = np.zeros((h, w + 2*clip_w, c), dtype=np.uint8)
        front_image_all[:, :clip_w, :] = fl_img[:,:clip_w,:]
        front_image_all[:,clip_w:clip_w+w, :] = front_img
        front_image_all[:,clip_w+w:, :] = fr_img[:,w-clip_w:,:]

            
        cv2.imwrite(os.path.join(self.save_front_img_path, '%06d.png'%self.start_index), front_image_all)
        img_time = time.time()
        # print("img time %.4f"%(img_time-raster_time))     

        ####### lidar raw data #######
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
        
        ####### vehicle CAN #######
        ego_vel = can_msg.VS_CAN # km/h
        ego_steer_ang = can_msg.mdps_str_ang # angle

        ####### track bboxes #######
        trk_features = []
        for trk in trk_bboxes_msg.objects:
            position = trk.pose.position
            dimensions = trk.dimensions
            quaternion = (
                trk.pose.orientation.x,
                trk.pose.orientation.y,
                trk.pose.orientation.z,
                trk.pose.orientation.w)
            euler = tf.transformations.euler_from_quaternion(quaternion)
            yaw = euler[2]
            feature = [self.start_index, trk.id, int(trk.label), position.x, position.y, position.z, dimensions.x, dimensions.y, dimensions.z, yaw, ego_vel, ego_steer_ang]
            trk_features.append(feature)
        # frame_id, object_id, object_type, position_x, position_y, position_z, object_length, object_width, object_height, heading, ego_vel, ego_steer_ang
        trk_features = np.array(trk_features)
        if len(trk_features.shape) == 1:
            trk_features = np.expand_dims(trk_features, axis=0)

        df = pd.DataFrame(trk_features, columns=cols_name)
        csv_file = os.path.join(save_base_dir, "agents_status.csv")
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
        test.header = velo_msg.header
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
        make_bev.save_dataset(make_bev.ego_pose_msg, make_bev.local_map_msg, make_bev.velo_msg, make_bev.front_img_msg, make_bev.fl_img_msg, make_bev.fr_img_msg, make_bev.trk_bboxes_msg, make_bev.can_msg)
        r.sleep()
    rospy.spin()