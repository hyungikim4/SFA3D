import os
import glob
import numpy as np
import math

base_dir = '/media/khg/5E103DDF103DBF39/veloster_rosbag/veloster_tracking_dataset/2020-09-24-17-53-51'
tram_ego_pose_dir = os.path.join(base_dir, 'ego_pose')
veloster_ego_pose_dir = os.path.join(base_dir, 'ego_pose_re')


save_dir = os.path.join(base_dir, 'ego_pose_veloster')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

tram_ego_pose_filelist = sorted(glob.glob(os.path.join(tram_ego_pose_dir, '*.txt')))
veloster_ego_pose_filelist = sorted(glob.glob(os.path.join(veloster_ego_pose_dir, '*.txt')))

for tram_ego_file in tram_ego_pose_filelist:
    ego_filename = os.path.basename(tram_ego_file)
    timestamp_sec = 0
    timestamp_nsec = 0
    with open(tram_ego_file, 'r') as f:
        line = f.readline()
        str_values = line.split(' ')
        timestamp_sec = float(str_values[0])
        timestamp_nsec = float(str_values[1])
    
    closest_time_stamp_sec = 100000
    closest_time_stamp_nsec = 100000
    closest_time_stamp_idx = -1
    for velo_ego_idx, veloster_ego_file in enumerate(veloster_ego_pose_filelist):
        with open(veloster_ego_file, 'r') as f:
            line = f.readline()
            str_values = line.split(' ')
            timestamp_sec_tmp = float(str_values[0])
            timestamp_nsec_tmp = float(str_values[1])

            diff_sec = abs(timestamp_sec-timestamp_sec_tmp)
            diff_nsec = abs(timestamp_nsec-timestamp_nsec_tmp)

            if diff_sec < closest_time_stamp_sec:
                closest_time_stamp_idx = velo_ego_idx
                closest_time_stamp_sec = diff_sec
                closest_time_stamp_nsec = diff_nsec
            elif diff_sec == closest_time_stamp_sec:
                if diff_nsec < closest_time_stamp_nsec:
                    closest_time_stamp_idx = velo_ego_idx
                    closest_time_stamp_nsec = diff_nsec
    if closest_time_stamp_idx != -1:
        with open(os.path.join(save_dir, ego_filename), 'w') as new_f:
            with open(veloster_ego_pose_filelist[closest_time_stamp_idx], 'r') as f:
                line = f.readline()
                new_f.write(line)
    print(ego_filename, closest_time_stamp_idx)
                
