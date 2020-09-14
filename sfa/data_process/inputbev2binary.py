import cv2
import os
import sys
import numpy as np

def rgbMap2binary(RGB_Map):
    RGB_Map[RGB_Map[:, :, 2]!=0] = np.array([255, 255, 255])
    return RGB_Map

ext = ".png"
rgb_map_dir_path = '/home/khg/Python_proj/SFA3D/dataset/veloster/training/front_bev'
rgb_map_list = os.listdir(rgb_map_dir_path)
rgb_map_list_ext = [file for file in rgb_map_list if file.endswith(ext)]
rgb_map_list_ext.sort()

for rgb_map_filename in rgb_map_list_ext:
    if (rgb_map_filename == '000083.png'):
        rgb_map_path = os.path.join(rgb_map_dir_path, rgb_map_filename)
        filtered_map_and_rgb_map = cv2.imread(rgb_map_path)
        h, w, c = filtered_map_and_rgb_map.shape
        w = int(w/2)
        rgb_map = filtered_map_and_rgb_map[:, w:, :]
        bin_map = rgbMap2binary(rgb_map)
        cv2.imshow('bin', bin_map)
        cv2.waitKey(0)