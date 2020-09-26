#!/usr/bin/env python3
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
import numpy as np
import timeit

import argparse
import sys
import os
import time
import warnings
import math
import glob

warnings.filterwarnings("ignore", category=UserWarning)

if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import torch

sys.path.append('./')
sys.path.append('/home/usrg/python_ws/SFA3D')
from sfa.models.model_utils import create_model
from sfa.utils.evaluation_utils import draw_predictions, convert_det_to_real_values
from sfa.data_process.transformation import lidar_to_camera_box
from sfa.utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from sfa.data_process.kitti_data_utils import Calibration
from sfa.utils.demo_utils import parse_demo_configs, do_detect, download_and_unzip, write_credit, do_detect_2sides
from sfa.data_process.veloster_2sides_bev_utils import makeBEVMap
# import sfa.config.kitti_config as cnf
import sfa.config.veloster_config_2sides as cnf
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

class SFA3D_inference():
    def __init__(self, data_root):
        self.conf_thres = 0.5
        configs = parse_demo_configs()
        configs.pretrained_path = '/home/usrg/python_ws/SFA3D/Model_veloster_608_608_epoch_1000.pth'
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

        self.save_bev_image = True
        self.bev_dir = os.path.join(data_root, 'bev')
        if self.save_bev_image and not os.path.exists(self.bev_dir):
            os.makedirs(self.bev_dir)
        self.results_dir = os.path.join(data_root, 'inference_results')

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        with open(os.path.join(self.results_dir, 'classes.txt'), 'w') as f:
            f.write("Pedestrian\n")
            f.write("Car\n")
            f.write("Cyclist\n")

        lidar_file_list = sorted(glob.glob(os.path.join(data_root, 'lidar/*.npy')))
        
        for lidar_file in lidar_file_list:
            bboxes_msg = self.inference_one(lidar_file)
            print(lidar_file)


    def inference_one(self, npy_file):
        save_txt_filename = os.path.basename(npy_file).replace(".npy", ".txt")
        gen_numpy = np.load(npy_file)
        lidar = get_filtered_lidar(gen_numpy, cnf.boundary)
        bev_map = makeBEVMap(lidar, cnf.boundary)
        if self.save_bev_image:
            bev_map_image = (np.transpose(bev_map, (1,2,0))*255).astype(np.uint8)
            bev_map_image = cv2.rotate(bev_map_image, cv2.ROTATE_180)
            save_bev_image_filename = os.path.basename(npy_file).replace(".npy", ".png")
            cv2.imwrite(os.path.join(self.bev_dir, save_bev_image_filename), bev_map_image)
        bev_map = torch.from_numpy(bev_map)
    
        with torch.no_grad():
            detections, bev_map, fps = do_detect(self.configs, self.model, bev_map, is_front=True)
        
        print(fps)

        bboxes_msg = BoundingBoxArray()
        
        results_txt = open(os.path.join(self.results_dir, save_txt_filename), 'w')

        for j in range(self.configs.num_classes):
            class_name = ID_TO_CLASS_NAME[j]

            if len(detections[j]) > 0:
                for det in detections[j]:
                    _score, _x, _y, _z, _h, _w, _l, _yaw = det
                    if (_score < self.conf_thres):
                        continue
                    yaw = -_yaw
                    x = _y / cnf.BEV_HEIGHT * cnf.bound_size_x + cnf.boundary_back['minX']
                    y = _x / cnf.BEV_WIDTH * cnf.bound_size_y + cnf.boundary['minY']
                    z = _z + cnf.boundary['minZ']
                    w = _w / cnf.BEV_WIDTH * cnf.bound_size_y
                    l = _l / cnf.BEV_HEIGHT * cnf.bound_size_x

                    pixel_x = -y/cnf.DISCRETIZATION + cnf.BEV_WIDTH/2.
                    pixel_y = cnf.BEV_HEIGHT/2. - x/cnf.DISCRETIZATION
                    # class, x, y, w, h, yaw
                    data = "%d %f %f %f %f %f\n"%(j,pixel_x,pixel_y,_w,_l,_yaw)
                    results_txt.write(data)
                    
                    # Bounding boxes
                    bbox = BoundingBox()

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

        results_txt.close()
        return bboxes_msg
        
if __name__ == '__main__':
    sfa3d = SFA3D_inference(data_root='/home/usrg/python_ws/SFA3D/dataset/veloster_2sides/prediction_dataset/2020-09-22-16-08-48')
