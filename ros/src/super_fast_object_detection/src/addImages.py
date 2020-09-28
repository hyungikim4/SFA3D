import cv2
import os
import glob

data_root = '/media/khg/Samsung_T5/lidar_object_detection/prediction_dataset/prediction_dataset/2020-09-24-17-40-26'
bev_dir = os.path.join(data_root, 'bev_sample')
map_dir = os.path.join(data_root, 'map')

bev_file_list = sorted(glob.glob(os.path.join(bev_dir, '*.png')))
for bev_file in bev_file_list:
    map_file = bev_file.replace("bev_sample", "map")
    bev_img = cv2.imread(bev_file)
    map_img = cv2.imread(map_file)

    bev_img = cv2.addWeighted(bev_img,0.8,map_img,0.2,0)
    cv2.imwrite(bev_file, bev_img)