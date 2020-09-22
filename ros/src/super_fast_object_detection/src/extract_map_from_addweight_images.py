import cv2
import glob
import os
import numpy as np

data_root = '/home/khg/Python_proj/SFA3D/dataset/veloster/training/2020-08-27-17-20-40_east'
front_bev_list = sorted(glob.glob(os.path.join(data_root, 'front_bev/*.png')))
back_bev_list = sorted(glob.glob(os.path.join(data_root, 'back_bev/*.png')))

map_dir = os.path.join(data_root, 'map')
if not os.path.exists(map_dir):
    os.makedirs(map_dir)

def extractBinaryMap(img_path):
    filename = os.path.basename(img_path)
    bev_img = cv2.imread(img_path)
    h, w, c = bev_img.shape
    bev_img = bev_img[:,:w/2,:]
    
    map_binary = np.zeros((h, w/2), dtype=np.uint8)
    map_binary[bev_img[:,:,1]==51] = 255
    return map_binary

def concatMap(front_img, back_img):
    h, w = front_img.shape
    img = np.zeros((2*h, w), dtype=np.uint8)
    img[:h,:] = front_img
    img[h:,:] = back_img
    return img

for front_path, back_path in zip(front_bev_list, back_bev_list):
    filename = os.path.basename(front_path)

    front_map_binary = extractBinaryMap(front_path)
    back_map_binary = extractBinaryMap(back_path)
    map_binary = concatMap(front_map_binary, back_map_binary)

    cv2.imwrite(os.path.join(map_dir, filename), map_binary)