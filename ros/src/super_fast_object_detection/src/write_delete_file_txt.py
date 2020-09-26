import os
import cv2
import glob

data_root = '/home/usrg/python_ws/SFA3D/dataset/veloster_2sides/prediction_dataset/2020-09-22-16-08-48'
bev_file_list = sorted(glob.glob(os.path.join(data_root, 'bev/*.png')))

f = open(os.path.join(data_root, 'delete.txt'), 'w')
for bev_file in bev_file_list:
    filename = os.path.basename(bev_file)
    bev_img = cv2.imread(bev_file)
    cv2.imshow(filename, bev_img)
    key = cv2.waitKey(0)

    if (key == 68 or key == 100):
        print(filename[:-4])
        f.write(filename[:-4]+"\n")
    elif (key == 27):
        break
    cv2.destroyAllWindows()
f.close()