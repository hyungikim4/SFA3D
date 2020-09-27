import os
import glob
import shutil
import sys


# data_root = '/media/khg/Samsung_T5/lidar_object_detection/prediction_dataset/prediction_dataset/2020-09-22-16-08-48'
# bev_dir = os.path.join(data_root, 'bev')
# bev_file_list = sorted(glob.glob(os.path.join(bev_dir, '*.png')))

# save_path = os.path.join(data_root, 'bev_sample')
# if not os.path.exists(save_path):
# 	os.makedirs(save_path)

# def copy_files(path, ori_filename, save_path, save_filename, ext):
# 	path_ = os.path.join(path,ori_filename+ext)
# 	save_path_ = os.path.join(save_path,save_filename+ext)
# 	shutil.copy(path_, save_path_)

# for i, bev_file in enumerate(bev_file_list):
#     if (i%10 == 0):
#         shutil.copy(bev_file, bev_file.replace('bev', 'bev_sample'))

data_root = '/media/khg/Samsung_T5/lidar_object_detection/prediction_dataset/prediction_dataset/2020-09-22-16-08-48'
bev_dir = os.path.join(data_root, 'bev_sample')
inference_dir = os.path.join(data_root, 'inference_results')
bev_file_list = sorted(glob.glob(os.path.join(bev_dir, '*.png')))

save_path = os.path.join(data_root, 'sample_label')
if not os.path.exists(save_path):
	os.makedirs(save_path)

for i, bev_file in enumerate(bev_file_list):
    inference = bev_file.replace("bev_sample", "inference_results").replace(".png", ".txt")
    shutil.copy(inference, inference.replace('inference_results', 'sample_label'))
shutil.copy(os.path.join(inference_dir, "classes.txt"), os.path.join(save_path, "classes.txt"))