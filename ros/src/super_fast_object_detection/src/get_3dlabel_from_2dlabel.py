import cv2
import glob
import os
import numpy as np
import veloster_config as cnf
from veloster_data_utils import get_filtered_lidar
from veloster_bev_utils import makeBEVMap, makeBEVMap_binary

prev_boundary = {
    "minX": 0.,
    "maxX": 40.,
    "minY": -20.,
    "maxY": 20.,
    "minZ": -0.50,
    "maxZ": 3.50
}

prev_bound_size_x = prev_boundary['maxX'] - prev_boundary['minX']
prev_bound_size_y = prev_boundary['maxY'] - prev_boundary['minY']
prev_bound_size_z = prev_boundary['maxZ'] - prev_boundary['minZ']

prev_boundary_back = {
    "minX": -40.,
    "maxX": 0.,
    "minY": -20.,
    "maxY": 20.,
    "minZ": -0.50,
    "maxZ": 3.50
}

prev_BEV_WIDTH = 416  # across y axis -25m ~ 25m
prev_BEV_HEIGHT = 416  # across x axis 0m ~ 50m
prev_DISCRETIZATION = (prev_boundary["maxX"] - prev_boundary["minX"]) / float(prev_BEV_HEIGHT)

# input: [cls_id, cx, cy, w, h, angle]
# output: [cls_id, x, y, z, height, width, length, ry]
def get_3dlabel_from_2dlabel(labels):
    object_labels = []
    for label in labels:
        [cls_id, cx, cy, w, h, angle] = label
        x = (prev_BEV_HEIGHT-cy)*prev_DISCRETIZATION
        y = -(cx - float(prev_BEV_HEIGHT)/2.)*prev_DISCRETIZATION
        if cls_id == 0 or cls_id == 2: # Pedestrian, Cyclist
            height = 1.73
        else: # Car
            height = 1.56
        z = height / 2.
        width = w*prev_DISCRETIZATION
        length = h*prev_DISCRETIZATION
        ry = -angle
        object_label = [cls_id, x, y, z, height, width, length, ry]
        object_labels.append(object_label)
    return object_labels
    
# input: [cls_id, x, y, z, height, width, length, ry]
# output: [cls_id, cx, cy, w, h, angle]
def get_2dlabel_from_3dlabel(labels):
    img_labels = []
    for label in labels:
        [cls_id, x, y, z, h, w, l, yaw] = label
        pixel_x = -y/cnf.DISCRETIZATION + cnf.BEV_WIDTH/2.
        pixel_y = cnf.BEV_HEIGHT - x/cnf.DISCRETIZATION
        pixel_w = w*cnf.BEV_WIDTH/cnf.bound_size_y
        pixel_h = l*cnf.BEV_HEIGHT/cnf.bound_size_x
        pixel_yaw = -yaw
        img_label = [cls_id, pixel_x, pixel_y, pixel_w, pixel_h, pixel_yaw]
        img_labels.append(img_label)
    return img_labels


def get_bev_img(lidar_file):
    gen_numpy = np.load(lidar_file)
    front_lidar = get_filtered_lidar(gen_numpy, cnf.boundary)
    back_lidar = get_filtered_lidar(gen_numpy, cnf.boundary_back)
    filtered_front_lidar = get_filtered_lidar(gen_numpy, cnf.boundary)
    filtered_back_lidar = get_filtered_lidar(gen_numpy, cnf.boundary_back)

    bev_map = makeBEVMap(front_lidar, cnf.boundary)
    bev_map_binary = makeBEVMap_binary(filtered_front_lidar, cnf.boundary)
    back_bevmap = makeBEVMap(back_lidar, cnf.boundary_back)
    back_bevmap_binary = makeBEVMap_binary(filtered_back_lidar, cnf.boundary_back)

    bev_map = np.transpose(bev_map, (1,2,0))
    back_bevmap = np.transpose(back_bevmap, (1,2,0))
    bev_map_binary = np.transpose(bev_map_binary, (1,2,0))
    back_bevmap_binary = np.transpose(back_bevmap_binary, (1,2,0))

    all_bev_map = np.zeros((cnf.BEV_HEIGHT, 2*cnf.BEV_WIDTH, 3))
    all_back_bev_map = np.zeros((cnf.BEV_HEIGHT, 2*cnf.BEV_WIDTH, 3))

    all_bev_map[:,:cnf.BEV_WIDTH,:] = bev_map
    all_bev_map[:,cnf.BEV_WIDTH:,:] = bev_map_binary

    all_back_bev_map[:,:cnf.BEV_WIDTH,:] = back_bevmap
    all_back_bev_map[:,cnf.BEV_WIDTH:,:] = back_bevmap_binary
    
    all_bev_map = (all_bev_map*255).astype(np.uint8)
    all_back_bev_map = (all_back_bev_map*255).astype(np.uint8)

    all_bev_map = cv2.rotate(all_bev_map, cv2.ROTATE_180)
    all_back_bev_map = cv2.rotate(all_back_bev_map, cv2.ROTATE_180)

    bev_map = all_bev_map
    back_bevmap = all_back_bev_map

    h, w, c = bev_map.shape
    bev_2side_map = np.zeros((2*h, w, c), dtype=np.uint8)
    bev_2side_map[:h,:,:] = bev_map
    bev_2side_map[h:,:,:] = back_bevmap

    return bev_2side_map

if __name__ == "__main__":
    data_root = '/home/usrg/python_ws/SFA3D/dataset/veloster/training/sdf/2020-08-27-17-20-40_east'
    label_list = sorted(glob.glob(os.path.join(data_root, 'front_label/*.txt')))
    lidar_path = os.path.join(data_root, 'lidar')
    new_bev_dir = os.path.join(data_root, 'bev')
    new_label_path = os.path.join(data_root, 'new_label')

    if not os.path.exists(new_bev_dir):
        os.makedirs(new_bev_dir)
    if not os.path.exists(new_label_path):
        os.makedirs(new_label_path)

    for label_file in label_list:
        filename = os.path.basename(label_file)
        if (filename == "classes.txt"):
            continue
        lidar_file = os.path.join(lidar_path, filename.replace(".txt", ".npy"))
        bev_2side_map = get_bev_img(lidar_file)

        img_labels = []
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                values = line.split(' ')
                cls_id = int(values[0])
                cx = float(values[1])
                cy = float(values[2])
                w = float(values[3])
                h = float(values[4])
                angle = float(values[5])
                label = [cls_id, cx, cy, w, h, angle]
                img_labels.append(label)
        lidar_labels = get_3dlabel_from_2dlabel(img_labels)
        new_img_labels = get_2dlabel_from_3dlabel(lidar_labels)
        
        with open(os.path.join(new_label_path, filename), 'w') as f:
            for new_label in new_img_labels:
                [cls_id, cx, cy, w, h, angle] = new_label
                data = "%d %f %f %f %f %f\n"%(cls_id, cx, cy, w, h, angle)
                f.write(data)
        
        cv2.imwrite(os.path.join(new_bev_dir, filename.replace(".txt", ".png")), bev_2side_map)
        print(os.path.join(new_label_path, filename))