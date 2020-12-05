#!/usr/bin/env python
import rosbag
import os
import numpy as np
import glob
import ros_numpy
import sensor_msgs
import tf

###### ref: timestamp
# base_dir = '/media/khg/5E103DDF103DBF39/veloster_rosbag/veloster_tracking_dataset/2020-09-22-16-08-48'
# ego_pose_dir = os.path.join(base_dir, 'ego_pose')
# ego_pose_filelist = sorted(glob.glob(os.path.join(ego_pose_dir, '*.txt')))

# filenames = []
# timestamp_sec = []
# timestamp_nsec = []
# for ego_file in ego_pose_filelist:
#     ego_idx = os.path.basename(ego_file)[:-4]
#     filenames.append(ego_idx)
#     with open(ego_file, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             str_values = line.split(' ')
#             timestamp_sec.append(float(str_values[0]))
#             timestamp_nsec.append(float(str_values[1]))

# filenames =  np.array(filenames)
# timestamp_sec = np.array(timestamp_sec)
# timestamp_nsec = np.array(timestamp_nsec)


# bag = rosbag.Bag('/media/khg/Elements/rosbag/200922_main_entrance/2020-09-22-16-08-48.bag')
# count = 0
# for topic, msg, t in bag.read_messages(topics=["/transformed_pointcloud", "/vehicle/pose"]):
#     if (topic == '/transformed_pointcloud'):
#         time_sec = msg.header.stamp.to_sec()
#         time_nsec = msg.header.stamp.to_nsec()
#         # index = np.where((timestamp_sec == time_sec) & (timestamp_nsec == time_nsec))
#         index = np.where((timestamp_sec == time_sec))
#         if (len(index[0]) != 0):
#             print(filenames[index])
#             print(time_sec, time_nsec, filenames[index])
    
#     # print(topic, msg.header.stamp.to_sec())
#     # print('-----')
#     count += 1
# print(count)
# bag.close()


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


base_dir = '/media/khg/5E103DDF103DBF39/veloster_rosbag/veloster_tracking_dataset/2020-09-24-17-40-26'
data_list_txt = os.path.join(base_dir, 'ImageSets/1/train.txt')
save_dir = os.path.join(base_dir, 'ego_pose_lidar_re')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

train_data_filelist = []
with open(data_list_txt, 'r') as f:
    lines = f.readlines()
    for line in lines:
        train_data_filelist.append(line.strip('\n'))

lidar_dir = os.path.join(base_dir, 'lidar')

bag_file = '/media/khg/Elements/rosbag/200924/2020-09-24-17-40-26.bag'
bag = rosbag.Bag(bag_file)

print(base_dir, bag_file)
count = 0

save_ego_pose = None
for topic, msg, t in bag.read_messages(topics=["/transformed_pointcloud", "/vehicle/pose"]):
    if (topic == '/transformed_pointcloud'):
        count += 1
        print(count)
        if len(train_data_filelist) == 0:
            print('finish')
            break
        msg.__class__ = sensor_msgs.msg._PointCloud2.PointCloud2
        ####### lidar raw data #######
        msg_numpy = ros_numpy.numpify(msg)
        if (len(msg_numpy) == 0):
            continue

        if (len(msg_numpy[0]) == 4): # if intensity field exists
            gen_numpy = get_xyzi_points(msg_numpy, remove_nans=True)
        else:
            xyz_array = ros_numpy.point_cloud2.get_xyz_points(msg_numpy, remove_nans=True)
            i = np.zeros((xyz_array.shape[0], 1))
            gen_numpy = np.concatenate((xyz_array,i), axis=1)

        for train_data_file in train_data_filelist:
            lidar_path = os.path.join(lidar_dir, train_data_file+'.npy')
            lidar_npy = np.load(lidar_path)
            
            if ((gen_numpy==lidar_npy).all()):
                print(t, train_data_file)
                save_ego_pose = train_data_file
                train_data_filelist.remove(save_ego_pose)
                break

    elif (topic == '/vehicle/pose'):
        if save_ego_pose is not None:
            time_sec = msg.header.stamp.to_sec()
            time_nsec = msg.header.stamp.to_nsec()

            ego_position = msg.pose.pose.position
            quaternion = (
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w)
            euler = tf.transformations.euler_from_quaternion(quaternion)
            ego_yaw = euler[2]

            data = "%f %f %f %f %f %f\n"%(time_sec, time_nsec, ego_position.x, ego_position.y, ego_position.z, ego_yaw)
            save_path = os.path.join(save_dir, save_ego_pose+'.txt')
            with open(os.path.join(save_dir, save_ego_pose+'.txt'), 'w') as f:
                f.write(data)
            print(save_path)
            save_ego_pose = None
        # # index = np.where((timestamp_sec == time_sec) & (timestamp_nsec == time_nsec))
        # index = np.where((timestamp_sec == time_sec))
        # if (len(index[0]) != 0):
        #     print(filenames[index])
        #     print(time_sec, time_nsec, filenames[index])
    
    # print(topic, msg.header.stamp.to_sec())
    # print('-----')
bag.close()
