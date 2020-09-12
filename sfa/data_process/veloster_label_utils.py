import sys
import os
import math
import numpy as np
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
sys.path.append('/home/khg/Python_proj/SFA3D')

from xml.etree.ElementTree import parse
import sfa.config.veloster_config as cnf

def getLabel(file_path):
    tree = parse(file_path)
    root = tree.getroot()

    objects = root.findall('object')
    names = [x.findtext('name') for x in objects]
    robndbox = [x.find('robndbox') for x in objects]
    bndbox = [x.find('bndbox') for x in objects]
    labels = []
    for obj in objects:
        name = obj.findtext('name')
        robndbox = obj.find('robndbox')
        if robndbox is None:
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.findtext('xmin'))
            ymin = float(bndbox.findtext('ymin'))
            xmax = float(bndbox.findtext('xmax'))
            ymax = float(bndbox.findtext('ymax'))

            label = [int(cnf.CLASS_NAME_TO_ID[name]), (xmin+xmax)/2., (ymin+ymax)/2., xmax-xmin, ymax-ymin, 0.0]
            labels.append(label)
            
        else:
            cx = float(robndbox.findtext('cx'))
            cy = float(robndbox.findtext('cy'))
            w = float(robndbox.findtext('w'))
            h = float(robndbox.findtext('h'))
            angle = float(robndbox.findtext('angle'))
            label = [int(cnf.CLASS_NAME_TO_ID[name]), cx, cy, w, h, angle]
            labels.append(label)

    if len(labels) == 0:
        labels = np.zeros((1, 6), dtype=np.float32)
        has_labels = False
    else:
        labels = np.array(labels, dtype=np.float32)
        has_labels = True
    return labels, has_labels

if __name__ == "__main__":
    labels, has_labels = getLabel('/home/khg/Python_proj/SFA3D/dataset/veloster/training/front_label/000030.xml')
    print(has_labels, labels)