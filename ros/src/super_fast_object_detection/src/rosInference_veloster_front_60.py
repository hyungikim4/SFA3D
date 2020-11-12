#!/usr/bin/env python3
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from autoware_msgs.msg import DetectedObjectArray, DetectedObject
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
import rospy
import rospkg
import numpy as np
import timeit

import argparse
import sys
import os
import time
import warnings
import zipfile
import ros_numpy
import math

warnings.filterwarnings("ignore", category=UserWarning)

if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import torch

sys.path.append('./')
sys.path.append('/home/usrg/catkin_ws/src/eurecar_ros/perception/environmental_perception/lidar/SFA3D')
from sfa.models.model_utils import create_model
from sfa.utils.evaluation_utils import draw_predictions, convert_det_to_real_values
from sfa.data_process.transformation import lidar_to_camera_box
from sfa.utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from sfa.data_process.kitti_data_utils import Calibration
from sfa.utils.demo_utils import parse_demo_configs, do_detect, download_and_unzip, write_credit, do_detect_2sides
from sfa.data_process.kitti_bev_utils import makeBEVMap
# import sfa.config.kitti_config as cnf
import sfa.config.veloster_config_front_60 as cnf
from sfa.data_process.kitti_data_utils import get_filtered_lidar

ID_TO_CLASS_NAME = {
    0: 'pedestrian',
    1: 'car',
    2: 'cyclist',
    -3: 'truck',
    -99: 'tram',
    -1: 'unknown'
}

scan = None
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

def euler_to_quaternion(yaw, pitch, roll):

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

def quaternion_to_euler(quat):
    x = quat[0]
    y = quat[1]
    z = quat[2]
    w = quat[3]

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw

class SFA3D():
    def __init__(self):
        self.is_callback = False
        self.scan = None
        self.conf_thres = 0.5
        # rospack = rospkg.RosPack()
        # package_path = rospack.get_path('super_fast_object_detection')
        configs = parse_demo_configs()
        # configs.pretrained_path = package_path + '/checkpoints/kitti_fpn_resnet_18/fpn_resnet_18_epoch_300.pth'
        configs.pretrained_path = '/home/usrg/catkin_ws/src/eurecar_ros/perception/environmental_perception/lidar/SFA3D/Model_veloster_608_608_3_epoch_1000.pth'
        model = create_model(configs)
        print('\n\n' + '-*=' * 30 + '\n\n')
        assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
        model.load_state_dict(torch.load(configs.pretrained_path, map_location='cuda:0'))
        print('Loaded weights from {}\n'.format(configs.pretrained_path))
        configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
        model = model.to(device=configs.device)
        self.model = model
        self.configs = configs
        self.model.eval()
        
        self.bboxes_pub = rospy.Publisher('/detection/bboxes', BoundingBoxArray, queue_size=1)
        self.detection_pub = rospy.Publisher('detected_objects', DetectedObjectArray, queue_size=1)
        
        self.velo_sub = rospy.Subscriber("/transformed_pointcloud", PointCloud2, self.velo_callback, queue_size=1, buff_size=2**24) # "/kitti/velo/pointcloud"
        print("Started Node")


    def velo_callback(self, msg):
        # print('callback')
        self.is_callback = True
        self.scan = msg
        self.on_scan(msg)
    
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
            roll, pitch, yaw = quaternion_to_euler(quaternion)

            rot_matrix = self.rot_z(yaw)
            rotated_corners_bev = np.dot(rot_matrix, corners_bev)
            corners_bev = rotated_corners_bev + xy
            bev_clusters_bboxes.append(corners_bev.transpose())
        return np.array(bev_clusters_bboxes)

    def on_scan(self, scan):
        if (scan is None or not self.is_callback):
            return
        if (scan is None):
            return
        start = timeit.default_timer()
        rospy.loginfo("Got scan")
        start_time = time.time()
        gen = []
        # for p in pc2.read_points(scan, field_names = ("x", "y", "z", "intensity"), skip_nans=True):
        #     gen.append(np.array([p[0], p[1], p[2], p[3]/100.0]))
        # gen_numpy = np.array(gen, dtype=np.float32)
        msg_numpy = ros_numpy.numpify(scan)
        if (len(msg_numpy) == 0):
            return

        if (len(msg_numpy[0]) == 4): # if intensity field exists
            gen_numpy = get_xyzi_points(msg_numpy, remove_nans=True)
        else:
            xyz_array = ros_numpy.point_cloud2.get_xyz_points(msg_numpy, remove_nans=True)
            i = np.zeros((xyz_array.shape[0], 1))
            gen_numpy = np.concatenate((xyz_array,i), axis=1)
        front_lidar = get_filtered_lidar(gen_numpy, cnf.boundary)
        bev_map = makeBEVMap(front_lidar, cnf.boundary)
        bev_map = torch.from_numpy(bev_map)
        back_lidar = get_filtered_lidar(gen_numpy, cnf.boundary_back)
        back_bevmap = makeBEVMap(back_lidar, cnf.boundary_back)
        back_bevmap = torch.from_numpy(back_bevmap)

        with torch.no_grad():
            detections, bev_map, fps = do_detect(self.configs, self.model, bev_map, is_front=True)
            # back_detections, back_bevmap, _ = do_detect(self.configs, self.model, back_bevmap, is_front=False)
            # detections, back_detections, bev_map, fps = do_detect_2sides(self.configs, self.model, bev_map, back_bevmap)
        print(fps)
        objects_msg = DetectedObjectArray()
        # objects_msg.header = scan.header
        objects_msg.header.stamp = rospy.Time.now()
        objects_msg.header.frame_id = scan.header.frame_id
        bboxes_msg = BoundingBoxArray()
        # bboxes_msg.header = scan.header
        bboxes_msg.header.stamp = rospy.Time.now()
        bboxes_msg.header.frame_id = scan.header.frame_id
        flag = False

        for j in range(self.configs.num_classes):
            class_name = ID_TO_CLASS_NAME[j]

            if len(detections[j]) > 0:
                # flag = True
                for det in detections[j]:
                    _score, _x, _y, _z, _h, _w, _l, _yaw = det
                    if (_score < self.conf_thres):
                        continue
                    yaw = -_yaw
                    x = _y / cnf.BEV_HEIGHT * cnf.bound_size_x + cnf.boundary['minX']
                    y = _x / cnf.BEV_WIDTH * cnf.bound_size_y + cnf.boundary['minY']
                    z = _z + cnf.boundary['minZ']
                    w = _w / cnf.BEV_WIDTH * cnf.bound_size_y
                    l = _l / cnf.BEV_HEIGHT * cnf.bound_size_x

                    # Autoware detected objects
                    obj = DetectedObject()
                    obj.header.stamp = rospy.Time.now()
                    obj.header.frame_id = scan.header.frame_id

                    obj.pose_reliable = True
                    obj.valid = True
                    
                    obj.space_frame = scan.header.frame_id
                    obj.label = class_name
                    obj.score = _score
                    obj.pose.position.x = x
                    obj.pose.position.y = y
                    obj.pose.position.z = z
                    [qx, qy, qz, qw] = euler_to_quaternion(yaw, 0, 0)
                    obj.pose.orientation.x = qx
                    obj.pose.orientation.y = qy
                    obj.pose.orientation.z = qz
                    obj.pose.orientation.w = qw
                    
                    obj.dimensions.x = l
                    obj.dimensions.y = w
                    obj.dimensions.z = _h
                    objects_msg.objects.append(obj)

                    # Bounding boxes
                    bbox = BoundingBox()
                    bbox.header.stamp = rospy.Time.now()
                    bbox.header.frame_id = scan.header.frame_id

                    bbox.label = j
                    bbox.pose.position.x = x
                    bbox.pose.position.y = y
                    bbox.pose.position.z = z
                    [qx, qy, qz, qw] = euler_to_quaternion(yaw, 0, 0)
                    bbox.pose.orientation.x = qx
                    bbox.pose.orientation.y = qy
                    bbox.pose.orientation.z = qz
                    bbox.pose.orientation.w = qw
                    
                    bbox.dimensions.x = l
                    bbox.dimensions.y = w
                    bbox.dimensions.z = _h
                    bboxes_msg.boxes.append(bbox)

            # if len(back_detections[j]) > 0:
            #     for det in back_detections[j]:
            #         _score, _x, _y, _z, _h, _w, _l, _yaw = det
            #         if (_score < self.conf_thres):
            #             continue
            #         yaw = -_yaw
            #         x = -_y / cnf.BEV_HEIGHT * cnf.bound_size_x + cnf.boundary_back['maxX']
            #         y = -_x / cnf.BEV_WIDTH * cnf.bound_size_y + cnf.boundary_back['maxY']
            #         z = _z + cnf.boundary_back['minZ']
            #         w = _w / cnf.BEV_WIDTH * cnf.bound_size_y
            #         l = _l / cnf.BEV_HEIGHT * cnf.bound_size_x
            #         obj = DetectedObject()
            #         obj.header.stamp = rospy.Time.now()
            #         obj.header.frame_id = scan.header.frame_id

            #         obj.pose_reliable = True
            #         obj.valid = True
                    
            #         obj.space_frame = scan.header.frame_id
            #         obj.label = class_name
            #         obj.score = _score
            #         obj.pose.position.x = x
            #         obj.pose.position.y = y
            #         obj.pose.position.z = z
            #         [qx, qy, qz, qw] = euler_to_quaternion(yaw, 0, 0)
            #         obj.pose.orientation.x = qx
            #         obj.pose.orientation.y = qy
            #         obj.pose.orientation.z = qz
            #         obj.pose.orientation.w = qw
                    
            #         obj.dimensions.x = l
            #         obj.dimensions.y = w
            #         obj.dimensions.z = _h
            #         objects_msg.objects.append(obj)

            #         # Bounding boxes
            #         bbox = BoundingBox()
            #         bbox.header.stamp = rospy.Time.now()
            #         bbox.header.frame_id = scan.header.frame_id

            #         bbox.label = j
            #         bbox.pose.position.x = x
            #         bbox.pose.position.y = y
            #         bbox.pose.position.z = z
            #         [qx, qy, qz, qw] = euler_to_quaternion(yaw, 0, 0)
            #         bbox.pose.orientation.x = qx
            #         bbox.pose.orientation.y = qy
            #         bbox.pose.orientation.z = qz
            #         bbox.pose.orientation.w = qw
                    
            #         bbox.dimensions.x = l
            #         bbox.dimensions.y = w
            #         bbox.dimensions.z = _h
            #         bboxes_msg.boxes.append(bbox)
        
        # raster agent mask
        # detection_time = time.time()
        # agents_mask = np.zeros((cnf.BEV_HEIGHT, cnf.BEV_WIDTH), dtype=np.uint8)

        # bev_bboxes = self.get_bev_bboxes(bboxes_msg)
        # for bev_bbox in bev_bboxes:    
        #     corner_2d = []
        #     for i in range(len(bev_bbox)):
        #         corner_2d.append(self.get_pixel_xy(bev_bbox[i][0], bev_bbox[i][1]))
        #     corner_2d = np.array(corner_2d)
        #     # print(corner_2d)
        #     corners_2d_dim = np.expand_dims(corner_2d, axis=1)
        #     corners_2d_dim = corners_2d_dim.astype(int)
        #     hull = cv2.convexHull(corners_2d_dim)
            
        #     cv2.drawContours(agents_mask, [hull], 0, (255, 0, 0), thickness=cv2.FILLED)
        # raster_time = time.time()
        # print('rasterization time: %.4f sec'%(raster_time-detection_time))
        # cv2.imshow('agents_mask', agents_mask)
        # cv2.waitKey(1)
        self.detection_pub.publish(objects_msg)
        self.bboxes_pub.publish(bboxes_msg)
            
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        self.is_callback = False
    


if __name__ == '__main__':
    rospy.init_node('SuperFastObjectDetection', anonymous=True)
    sfa3d = SFA3D()
    # while not rospy.is_shutdown():
    #     sfa3d.on_scan(sfa3d.scan)
    rospy.spin()
