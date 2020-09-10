#!/usr/bin/env python
import rospy, math, random
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from laser_geometry import LaserProjection
import sensor_msgs.point_cloud2 as pc2
import ros_numpy

from std_msgs.msg import String
import std_msgs.msg
#import matplotlib.pyplot as plt
import os.path
import os
import glob
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/home/khg/Python_proj/SFA3D')
from sfa.data_process.kitti_bev_utils import makeBEVMap, makeBEVMap_binary
import sfa.config.veloster_config as cnf
from sfa.data_process.kitti_data_utils import get_filtered_lidar


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
        self.do_overwrite = True
        self.velo_sub = rospy.Subscriber("/transformed_pointcloud", PointCloud2, self.velo_callback, queue_size=1)
        self.save_dir_path = '/home/khg/Python_proj/SFA3D/dataset/veloster/training'
        self.save_lidar_path = os.path.join(self.save_dir_path, 'lidar')
        self.save_front_bev_path = os.path.join(self.save_dir_path, 'front_bev')
        self.save_back_bev_path = os.path.join(self.save_dir_path, 'back_bev')
        self.save_front_label_path = os.path.join(self.save_dir_path, 'front_label')
        self.save_back_label_path = os.path.join(self.save_dir_path, 'back_label')
        if not os.path.exists(self.save_dir_path):
            os.mkdir(self.save_dir_path)
        if not os.path.exists(self.save_lidar_path):
            os.mkdir(self.save_lidar_path)
        if not os.path.exists(self.save_front_bev_path):
            os.mkdir(self.save_front_bev_path)
        if not os.path.exists(self.save_back_bev_path):
            os.mkdir(self.save_back_bev_path)
        if not os.path.exists(self.save_front_label_path):
            os.mkdir(self.save_front_label_path)
        if not os.path.exists(self.save_back_label_path):
            os.mkdir(self.save_back_label_path)
        lidar_file_list = os.listdir(self.save_lidar_path)
        lidar_file_list_npy = [file for file in lidar_file_list if file.endswith(".npy")]
        
        if self.do_overwrite:
            self.start_index = 0
        else:
            self.start_index = len(lidar_file_list_npy)
        print('start_index', self.start_index)
    def velo_callback(self, msg):
        # save lidar raw data
        msg_numpy = ros_numpy.numpify(msg)
        if (len(msg_numpy) == 0):
            return

        if (len(msg_numpy[0]) == 4): # if intensity field exists
            gen_numpy = get_xyzi_points(msg_numpy, remove_nans=True)
        else:
            xyz_array = ros_numpy.point_cloud2.get_xyz_points(msg_numpy, remove_nans=True)
            i = np.zeros((xyz_array.shape[0], 1))
            gen_numpy = np.concatenate((xyz_array,i), axis=1)
        
        np.save(os.path.join(self.save_lidar_path, '%06d'%self.start_index), gen_numpy)
        
        # save bev images
        front_lidar = get_filtered_lidar(gen_numpy, cnf.boundary)
        back_lidar = get_filtered_lidar(gen_numpy, cnf.boundary_back)
        if self.save_RGB_binray_all:
            bev_map = makeBEVMap(front_lidar, cnf.boundary)
            back_bevmap = makeBEVMap(back_lidar, cnf.boundary_back)
            bev_map_binary = makeBEVMap_binary(front_lidar, cnf.boundary)
            back_bevmap_binary = makeBEVMap_binary(back_lidar, cnf.boundary_back)

            bev_map = np.transpose(bev_map, (2,1,0))
            back_bevmap = np.transpose(back_bevmap, (2,1,0))
            bev_map_binary = np.transpose(bev_map_binary, (2,1,0))
            back_bevmap_binary = np.transpose(back_bevmap_binary, (2,1,0))

            all_bev_map = np.zeros((cnf.BEV_HEIGHT, 2*cnf.BEV_WIDTH, 3))
            all_back_bev_map = np.zeros((cnf.BEV_HEIGHT, 2*cnf.BEV_WIDTH, 3))

            all_bev_map[:,:cnf.BEV_WIDTH,:] = bev_map
            all_bev_map[:,cnf.BEV_WIDTH:,:] = bev_map_binary
            all_back_bev_map[:,:cnf.BEV_WIDTH,:] = back_bevmap
            all_back_bev_map[:,cnf.BEV_WIDTH:,:] = back_bevmap_binary

            bev_map = (all_bev_map*255).astype(np.uint8)
            back_bevmap = (all_back_bev_map*255).astype(np.uint8)
        else:
            if self.save_RGB_bev_input:
                bev_map = makeBEVMap(front_lidar, cnf.boundary)
                back_bevmap = makeBEVMap(back_lidar, cnf.boundary_back)
            else:
                bev_map = makeBEVMap_binary(front_lidar, cnf.boundary)
                back_bevmap = makeBEVMap_binary(back_lidar, cnf.boundary_back)
        
            bev_map = np.transpose(bev_map, (2,1,0))
            back_bevmap = np.transpose(back_bevmap, (2,1,0))
            bev_map = (bev_map*255).astype(np.uint8)
            back_bevmap = (back_bevmap*255).astype(np.uint8)
        # cv2.imshow('bev', bev_map)
        # cv2.imshow('back', back_bevmap)
        # cv2.waitKey(1)

        cv2.imwrite(os.path.join(self.save_front_bev_path, '%06d.png'%self.start_index), bev_map)
        cv2.imwrite(os.path.join(self.save_back_bev_path, '%06d.png'%self.start_index), back_bevmap)
        self.start_index += 1

if __name__=='__main__':
    rospy.init_node('make_bin_and_bev_images', anonymous=True)
    make_bev = MakeBevImages()
    rospy.spin()