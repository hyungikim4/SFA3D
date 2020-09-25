"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.17
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for the KITTI dataset
"""
import sys
import os
import math
from builtins import int

import numpy as np
from torch.utils.data import Dataset
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
sys.path.append('/home/khg/Python_proj/SFA3D')
import cv2
import torch

sys.path.append('../')

from sfa.data_process.kitti_data_utils import gen_hm_radius, compute_radius, Calibration, get_filtered_lidar
from sfa.data_process.veloster_2sides_bev_utils import makeBEVMap, drawRotatedBox, get_corners
from sfa.data_process import transformation

from xml.etree.ElementTree import parse
import sfa.config.veloster_config_2sides as cnf

class VelosterDataset(Dataset):
    def __init__(self, configs, mode='train', lidar_aug=None, hflip_prob=None, num_samples=None):
        self.dataset_dir = configs.dataset_dir
        self.input_size = configs.input_size
        self.hm_size = configs.hm_size

        self.num_classes = configs.num_classes
        self.max_objects = configs.max_objects

        assert mode in ['train', 'val', 'test'], 'Invalid mode: {}'.format(mode)
        self.mode = mode
        self.is_test = (self.mode == 'test')
        sub_folder = 'testing' if self.is_test else 'training'

        self.lidar_aug = lidar_aug
        self.hflip_prob = hflip_prob

        self.image_dir = os.path.join(self.dataset_dir, sub_folder, "front_image")
        self.lidar_dir = os.path.join(self.dataset_dir, sub_folder, "lidar")
        self.calib_dir = os.path.join(self.dataset_dir, sub_folder, "calib")
        self.label_dir = os.path.join(self.dataset_dir, sub_folder, "label")
        split_txt_path = os.path.join(self.dataset_dir, 'ImageSets', '{}.txt'.format(mode))
        self.sample_id_list = [int(x.strip()) for x in open(split_txt_path).readlines()]

        if num_samples is not None:
            self.sample_id_list = self.sample_id_list[:num_samples]
        self.num_samples = len(self.sample_id_list)

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, index):
        if self.is_test:
            return self.load_img_only(index)
        else:
            return self.load_img_with_targets(index)

    def load_img_only(self, index):
        """Load only image for the testing phase"""
        sample_id = int(self.sample_id_list[index])
        img_path, img_rgb = self.get_image(sample_id)
        lidarData = self.get_lidar(sample_id)
        lidarData = get_filtered_lidar(lidarData, cnf.boundary)
        bev_map = makeBEVMap(lidarData, cnf.boundary)
        bev_map = torch.from_numpy(bev_map)

        metadatas = {
            'img_path': img_path,
        }

        return metadatas, bev_map, img_rgb

    def load_img_with_targets(self, index):
        """Load images and targets for the training and validation phase"""
        sample_id = int(self.sample_id_list[index])
        img_path = os.path.join(self.image_dir, '{:06d}.png'.format(sample_id))
        lidarData = self.get_lidar(sample_id)
        # calib = self.get_calib(sample_id)
        # [cat_id, x, y, z, h, w, l, ry]
        labels, has_labels = self.get_label(sample_id)
        # if has_labels:
        #     labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0, calib.P2)

        if self.lidar_aug:
            lidarData, labels[:, 1:] = self.lidar_aug(lidarData, labels[:, 1:])

        lidarData, labels = get_filtered_lidar(lidarData, cnf.boundary, labels)

        bev_map = makeBEVMap(lidarData, cnf.boundary)
        bev_map = torch.from_numpy(bev_map)

        hflipped = False
        if np.random.random() < self.hflip_prob:
            hflipped = True
            # C, H, W
            bev_map = torch.flip(bev_map, [-1])

        targets = self.build_targets(labels, hflipped)

        metadatas = {
            'img_path': img_path,
            'hflipped': hflipped
        }

        return metadatas, bev_map, targets

    def get_image(self, idx):
        img_path = os.path.join(self.image_dir, '{:06d}.png'.format(idx))
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        return img_path, img

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '{:06d}.txt'.format(idx))
        # assert os.path.isfile(calib_file)
        return Calibration(calib_file)

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '{:06d}.npy'.format(idx))
        # assert os.path.isfile(lidar_file)
        # return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        return np.load(lidar_file).reshape(-1, 4)

    # https://github.com/zexihan/labelImg-kitti
    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '{:06d}.txt'.format(idx))

        f = open(label_file)
        lines = f.readlines()
        f.close()
        
        labels = []
        img_labels = []
        for line in lines:
            string_values = line.split(' ')
            cls_id = int(string_values[0])
            cx = float(string_values[1])
            cy = float(string_values[2])
            w = float(string_values[3])
            h = float(string_values[4])
            yaw = float(string_values[5])

            if (cx >= cnf.BEV_WIDTH):
                cx -= cnf.BEV_WIDTH
            label = [cls_id, cx, cy, w, h, yaw]
            img_labels.append(label)
        
        # Convert from image coodi to lidar coodi
        for img_label in img_labels:
            [cls_id, cx, cy, w, h, angle] = img_label
            x = (cnf.BEV_HEIGHT-cy)*cnf.DISCRETIZATION + cnf.boundary["minX"]
            y = -(cx - float(cnf.BEV_HEIGHT)/2.)*cnf.DISCRETIZATION
            if cls_id == 0 or cls_id == 2: # Pedestrian, Cyclist
                height = 1.73
            else: # Car
                height = 1.56
            z = height / 2.
            width = w*cnf.DISCRETIZATION
            length = h*cnf.DISCRETIZATION
            ry = -angle
            object_label = [cls_id, x, y, z, height, width, length, ry]
            labels.append(object_label)

        if len(labels) == 0:
            labels = np.zeros((1, 8), dtype=np.float32)
            has_labels = False
        else:
            labels = np.array(labels, dtype=np.float32)
            has_labels = True
        return labels, has_labels
    
    # # https://github.com/cgvict/roLabelImg
    # def get_label(self, idx):
    #     label_file = os.path.join(self.label_dir, '{:06d}.xml'.format(idx))
    #     tree = parse(label_file)
    #     root = tree.getroot()

    #     objects = root.findall('object')
    #     names = [x.findtext('name') for x in objects]
    #     robndbox = [x.find('robndbox') for x in objects]
    #     bndbox = [x.find('bndbox') for x in objects]
    #     img_labels = []
    #     labels = []
    #     for obj in objects:
    #         name = obj.findtext('name')
    #         robndbox = obj.find('robndbox')
    #         if robndbox is None:
    #             bndbox = obj.find('bndbox')
    #             xmin = float(bndbox.findtext('xmin'))
    #             ymin = float(bndbox.findtext('ymin'))
    #             xmax = float(bndbox.findtext('xmax'))
    #             ymax = float(bndbox.findtext('ymax'))
    #             if (xmin >= cnf.BEV_WIDTH):
    #                 xmin -= cnf.BEV_WIDTH
    #                 xmax -= cnf.BEV_WIDTH
    #             label = [int(cnf.CLASS_NAME_TO_ID[name]), (xmin+xmax)/2., (ymin+ymax)/2., xmax-xmin, ymax-ymin, 0.0]
    #             img_labels.append(label)
                
    #         else:
    #             cx = float(robndbox.findtext('cx'))
    #             cy = float(robndbox.findtext('cy'))
    #             w = float(robndbox.findtext('w'))
    #             h = float(robndbox.findtext('h'))
    #             if (cx-w/2. >= cnf.BEV_WIDTH):
    #                 cx -= cnf.BEV_WIDTH
    #             angle = float(robndbox.findtext('angle'))
    #             label = [int(cnf.CLASS_NAME_TO_ID[name]), cx, cy, w, h, angle]
    #             img_labels.append(label)
        
    #     # Convert from image coodi to lidar coodi
    #     for img_label in img_labels:
    #         [cls_id, cx, cy, w, h, angle] = img_label
    #         x = (cnf.BEV_HEIGHT-cy)*cnf.DISCRETIZATION
    #         y = -(cx - float(cnf.BEV_HEIGHT)/2.)*cnf.DISCRETIZATION
    #         if cls_id == 0 or cls_id == 2: # Pedestrian, Cyclist
    #             height = 1.73
    #         else: # Car
    #             height = 1.56
    #         z = height / 2.
    #         width = w*cnf.DISCRETIZATION
    #         length = h*cnf.DISCRETIZATION
    #         ry = -angle
    #         object_label = [cls_id, x, y, z, height, width, length, ry]
    #         labels.append(object_label)

    #     if len(labels) == 0:
    #         labels = np.zeros((1, 8), dtype=np.float32)
    #         has_labels = False
    #     else:
    #         labels = np.array(labels, dtype=np.float32)
    #         has_labels = True
    #     return labels, has_labels

    def build_targets(self, labels, hflipped):
        minX = cnf.boundary['minX']
        maxX = cnf.boundary['maxX']
        minY = cnf.boundary['minY']
        maxY = cnf.boundary['maxY']
        minZ = cnf.boundary['minZ']
        maxZ = cnf.boundary['maxZ']

        num_objects = min(len(labels), self.max_objects)
        hm_l, hm_w = self.hm_size

        hm_main_center = np.zeros((self.num_classes, hm_l, hm_w), dtype=np.float32)
        cen_offset = np.zeros((self.max_objects, 2), dtype=np.float32)
        direction = np.zeros((self.max_objects, 2), dtype=np.float32)
        z_coor = np.zeros((self.max_objects, 1), dtype=np.float32)
        dimension = np.zeros((self.max_objects, 3), dtype=np.float32)

        indices_center = np.zeros((self.max_objects), dtype=np.int64)
        obj_mask = np.zeros((self.max_objects), dtype=np.uint8)

        for k in range(num_objects):
            cls_id, x, y, z, h, w, l, yaw = labels[k]
            cls_id = int(cls_id)
            # Invert yaw angle
            yaw = -yaw
            if not ((minX <= x <= maxX) and (minY <= y <= maxY) and (minZ <= z <= maxZ)):
                continue
            if (h <= 0) or (w <= 0) or (l <= 0):
                continue

            bbox_l = l / cnf.bound_size_x * hm_l
            bbox_w = w / cnf.bound_size_y * hm_w
            radius = compute_radius((math.ceil(bbox_l), math.ceil(bbox_w)))
            radius = max(0, int(radius))

            center_y = (x - minX) / cnf.bound_size_x * hm_l  # x --> y (invert to 2D image space)
            center_x = (y - minY) / cnf.bound_size_y * hm_w  # y --> x
            center = np.array([center_x, center_y], dtype=np.float32)

            if hflipped:
                center[0] = hm_w - center[0] - 1

            center_int = center.astype(np.int32)
            if cls_id < 0:
                ignore_ids = [_ for _ in range(self.num_classes)] if cls_id == - 1 else [- cls_id - 2]
                # Consider to make mask ignore
                for cls_ig in ignore_ids:
                    gen_hm_radius(hm_main_center[cls_ig], center_int, radius)
                hm_main_center[ignore_ids, center_int[1], center_int[0]] = 0.9999
                continue

            # Generate heatmaps for main center
            gen_hm_radius(hm_main_center[cls_id], center, radius)
            # Index of the center
            indices_center[k] = center_int[1] * hm_w + center_int[0]

            # targets for center offset
            cen_offset[k] = center - center_int

            # targets for dimension
            dimension[k, 0] = h
            dimension[k, 1] = w
            dimension[k, 2] = l

            # targets for direction
            direction[k, 0] = math.sin(float(yaw))  # im
            direction[k, 1] = math.cos(float(yaw))  # re
            # im -->> -im
            if hflipped:
                direction[k, 0] = - direction[k, 0]

            # targets for depth
            z_coor[k] = z - minZ

            # Generate object masks
            obj_mask[k] = 1
            
        targets = {
            'hm_cen': hm_main_center,
            'cen_offset': cen_offset,
            'direction': direction,
            'z_coor': z_coor,
            'dim': dimension,
            'indices_center': indices_center,
            'obj_mask': obj_mask,
        }

        return targets

    def draw_img_with_label(self, index):
        sample_id = int(self.sample_id_list[index])
        img_path, img_rgb = self.get_image(sample_id)
        lidarData = self.get_lidar(sample_id)
        # calib = self.get_calib(sample_id)
        labels, has_labels = self.get_label(sample_id)

        if self.lidar_aug:
            lidarData, labels[:, 1:] = self.lidar_aug(lidarData, labels[:, 1:])

        lidarData = get_filtered_lidar(lidarData, cnf.boundary)
        bev_map = makeBEVMap(lidarData, cnf.boundary)

        return bev_map, labels, img_rgb, img_path




if __name__ == '__main__':
    from easydict import EasyDict as edict
    from data_process.transformation import OneOf, Random_Scaling, Random_Rotation, lidar_to_camera_box
    from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes, show_rgb_image_with_boxes_matrix

    # ROS
    sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import rospy
    from visualization_msgs.msg import MarkerArray, Marker
    from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
    from sensor_msgs.msg import PointCloud2, PointField
    import sensor_msgs.point_cloud2 as pc2

    def euler_to_quaternion(roll, pitch, yaw):
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

    def npy2pointcloud2_msg(cloud, stamp, frame_id):
        pc2_msg = PointCloud2()
        pc2_msg.header.stamp = stamp
        pc2_msg.header.frame_id = frame_id
        lidar = []
        for i in range(cloud.shape[0]):
            lidar.append(list(cloud[i,:3]))
        pc2_msg2 = pc2.create_cloud_xyz32(pc2_msg.header, lidar)
        return pc2_msg2

    def labels2Marker_msg(labels, stamp, frame_id):
        marker_id = 0
        obj_marker_arr = MarkerArray()
        for label in labels:
            cls_id, x, y, z, h, w, l, yaw = label
            obj_marker = Marker()
            obj_marker.header.stamp = stamp
            obj_marker.header.frame_id = frame_id
            obj_marker.lifetime = rospy.Duration(1.)
            obj_marker.type = Marker.MESH_RESOURCE
            obj_marker.action = Marker.ADD
            obj_marker.mesh_use_embedded_materials = False
            obj_marker.pose.position.x = x
            obj_marker.pose.position.y = y
            obj_marker.pose.position.z = z
            [qx,qy,qz,qw] = euler_to_quaternion(0,0,yaw)
            obj_marker.pose.orientation.x = qx
            obj_marker.pose.orientation.y = qy
            obj_marker.pose.orientation.z = qz
            obj_marker.pose.orientation.w = qw

            obj_marker.color.r = 0.
            obj_marker.color.g = 1.
            obj_marker.color.b = 0.
            obj_marker.color.a = 0.3
            obj_marker.id = marker_id

            obj_marker.scale.x = 1
            obj_marker.scale.y = 1
            obj_marker.scale.z = 1
            obj_marker.mesh_resource = "file:///home/khg/Python_proj/GRIP/dae_models/box.dae"

            marker_id += 1
            obj_marker_arr.markers.append(obj_marker)
        return obj_marker_arr

    def labels2Bboxes_msg(labels, stamp, frame_id):
        bboxes = BoundingBoxArray()
        bboxes.header.stamp = stamp
        bboxes.header.frame_id = frame_id
        for label in labels:
            cls_id, x, y, z, h, w, l, yaw = label
            bbox = BoundingBox()
            bbox.header.stamp = stamp
            bbox.header.frame_id = frame_id
            
            bbox.pose.position.x = x
            bbox.pose.position.y = y
            bbox.pose.position.z = z
            [qx,qy,qz,qw] = euler_to_quaternion(0,0,yaw)
            bbox.pose.orientation.x = qx
            bbox.pose.orientation.y = qy
            bbox.pose.orientation.z = qz
            bbox.pose.orientation.w = qw

            bbox.dimensions.x = l
            bbox.dimensions.y = w
            bbox.dimensions.z = h
            bbox.label = int(cls_id)

            bboxes.boxes.append(bbox)
        return bboxes
            
    rospy.init_node('veloster_dataset')
    bboxes_pub = rospy.Publisher("/detection/bboxes", BoundingBoxArray, queue_size=1)
    # detection_marker_pub = rospy.Publisher("/detection/markers", MarkerArray, queue_size=1)
    lidar_pub = rospy.Publisher("/velodyne_points", PointCloud2, queue_size=1)
    rate = rospy.Rate(10) # 10hz

    configs = edict()
    configs.distributed = False  # For testing
    configs.pin_memory = False
    configs.num_samples = None
    configs.input_size = (608, 608)
    configs.hm_size = (152, 152)
    configs.max_objects = 50
    configs.num_classes = 3
    configs.output_width = 608

    configs.dataset_dir = os.path.join('../../', 'dataset', 'veloster_2sides')
    # lidar_aug = OneOf([
    #     Random_Rotation(limit_angle=np.pi / 4, p=1.),
    #     Random_Scaling(scaling_range=(0.95, 1.05), p=1.),
    # ], p=1.)
    lidar_aug = None

    dataset = VelosterDataset(configs, mode='train', lidar_aug=lidar_aug, hflip_prob=0., num_samples=configs.num_samples)
    
    print('\n\nPress n to see the next sample >>> Press Esc to quit...')
    for idx in range(len(dataset)):
        # if (idx != 0):
        #     continue
        bev_map, labels, img_rgb, img_path = dataset.draw_img_with_label(idx)
        print(img_path)
        img_h, img_w, img_c = img_rgb.shape
        img_w_uint = int(img_w/7)
        img_rgb = img_rgb[:,img_w_uint*2:img_w_uint*5,:]
        bev_map = (bev_map.transpose(1, 2, 0) * 255).astype(np.uint8)
        bev_map = cv2.resize(bev_map, (cnf.BEV_HEIGHT, cnf.BEV_WIDTH))

        lidar_path = img_path.replace(".png", ".npy").replace("front_image", "lidar")
        lidar_cloud = np.load(lidar_path)
        lidar_msg = npy2pointcloud2_msg(lidar_cloud, rospy.Time.now(), 'base_link')
        # markers_msg = labels2Marker_msg(labels, rospy.Time.now(), 'base_link')
        bboxes_msg = labels2Bboxes_msg(labels, rospy.Time.now(), 'base_link')
        for box_idx, (cls_id, x, y, z, h, w, l, yaw) in enumerate(labels):
            # Draw rotated box
            yaw = -yaw
            y1 = int((x - cnf.boundary['minX']) / cnf.DISCRETIZATION)
            x1 = int((y - cnf.boundary['minY']) / cnf.DISCRETIZATION)
            w1 = int(w / cnf.DISCRETIZATION)
            l1 = int(l / cnf.DISCRETIZATION)

            drawRotatedBox(bev_map, x1, y1, w1, l1, yaw, cnf.colors[int(cls_id)])
        # Rotate the bev_map
        bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_rgb = show_rgb_image_with_boxes_matrix(img_rgb, labels, cnf.Tr_velo_to_cam)

        out_img = merge_rgb_to_bev(img_rgb, bev_map, output_width=configs.output_width)
        cv2.imshow('bev_map', out_img)
        
        lidar_pub.publish(lidar_msg)
        # detection_marker_pub.publish(markers_msg)
        bboxes_pub.publish(bboxes_msg)
        if cv2.waitKey(0) & 0xff == 27:
            break
