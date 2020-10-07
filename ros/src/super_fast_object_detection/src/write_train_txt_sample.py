import sys
import os
import glob

sample_dir = '/media/usrg/Samsung_T5/lidar_object_detection/prediction_dataset/prediction_dataset/2020-09-24-17-53-51/bev_sample/1'
txt_path = '/media/usrg/Samsung_T5/lidar_object_detection/prediction_dataset/prediction_dataset/2020-09-24-17-53-51/ImageSets/1'

if not os.path.exists(txt_path):
    os.makedirs(txt_path)

ext = ".png"
sample_list = os.listdir(sample_dir)
sample_list_ext = [file for file in sample_list if file.endswith(ext)]
sample_list_ext.sort()

with open(os.path.join(txt_path, 'train.txt'), 'w') as f:
    for sample_file in sample_list_ext:
        f.write(sample_file[:-4]+"\n")