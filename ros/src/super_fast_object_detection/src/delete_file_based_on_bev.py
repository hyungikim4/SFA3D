import os
import cv2
import glob

data_root = '/home/usrg/python_ws/SFA3D/dataset/veloster_2sides/prediction_dataset/2020-09-22-16-08-48'
bev_file_list = sorted(glob.glob(os.path.join(data_root, 'bev/*.png')))
lidar_dir = os.path.join(data_root, 'lidar')
inference_dir = os.path.join(data_root, 'inference_results')
image_dir = os.path.join(data_root, 'front_image')
map_dir = os.path.join(data_root, 'map')

f = open(os.path.join(data_root, 'train.txt'), 'w')
for bev_file in bev_file_list:
    filename = os.path.basename(bev_file)
    f.write(filename[:-4]+"\n")
