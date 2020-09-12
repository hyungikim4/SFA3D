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
    labels = []
    for name, bbox in zip(names, robndbox):
        cx = float(bbox.findtext('cx'))
        cy = float(bbox.findtext('cy'))
        w = float(bbox.findtext('w'))
        h = float(bbox.findtext('h'))
        angle = float(bbox.findtext('angle'))
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
    labels, has_labels = getLabel('/home/khg/Python_proj/SFA3D/dataset/veloster/training/front_label/000541.xml')
    print(has_labels, labels)
