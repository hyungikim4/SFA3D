import sys
import os
import glob

st = 2668
end = 3378

txt_path = '/media/usrg/Samsung_T5/lidar_object_detection/prediction_dataset/prediction_dataset/2020-09-24-17-53-51/ImageSets/2'
with open(os.path.join(txt_path, 'train.txt'), 'w') as f:
    for i in range(st, end+1):
        f.write("%06d\n"%(i))