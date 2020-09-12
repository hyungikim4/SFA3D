import os
import glob
import shutil
import sys

ext = ".xml"

# path = sys.argv[1]

path = "/home/khg/Python_proj/SFA3D/dataset/veloster/training/"
save_path = "/home/khg/Python_proj/SFA3D/dataset/veloster/ImageSets/"
if not os.path.exists(save_path):
	os.makedirs(save_path)
label_path = os.path.join(path,"front_label")

f = open(os.path.join(save_path, "train.txt"), 'w')

label_list = os.listdir(label_path)
label_list_ext = [file for file in label_list if file.endswith(ext)]
label_list_ext.sort()

def copy_files(path, ori_filename, save_path, save_filename, ext):
	path_ = os.path.join(path,ori_filename+ext)
	save_path_ = os.path.join(save_path,save_filename+ext)
	shutil.copy(path_, save_path_)

for num, label_file in enumerate(label_list_ext):
	filename = label_file[:-4]
	
	data = filename+"\n"
	f.write(data)
	
f.close()

# copy_files(save_path, "train", save_path, "test",".txt")
