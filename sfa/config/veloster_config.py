import math

import numpy as np

# Car and Van ==> Car class
# Pedestrian and Person_Sitting ==> Pedestrian Class
CLASS_NAME_TO_ID = {
    'Pedestrian': 0,
    'Car': 1,
    'Cyclist': 2,
    'Van': 1,
    'Truck': -3,
    'Person_sitting': 0,
    'Tram': -99,
    'Misc': -99,
    'DontCare': -1
}

colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0], [255, 120, 0],
          [255, 120, 120], [0, 120, 0], [120, 255, 255], [120, 0, 255]]

#####################################################################################
# boundary = {
#     "minX": 0,
#     "maxX": 50,
#     "minY": -25,
#     "maxY": 25,
#     "minZ": -0.50,
#     "maxZ": 3.50
# }

# bound_size_x = boundary['maxX'] - boundary['minX']
# bound_size_y = boundary['maxY'] - boundary['minY']
# bound_size_z = boundary['maxZ'] - boundary['minZ']

# boundary_back = {
#     "minX": -50,
#     "maxX": 0,
#     "minY": -25,
#     "maxY": 25,
#     "minZ": -0.50,
#     "maxZ": 3.50
# }

# BEV_WIDTH = 608  # across y axis -25m ~ 25m
# BEV_HEIGHT = 608  # across x axis 0m ~ 50m
# DISCRETIZATION = (boundary["maxX"] - boundary["minX"]) / BEV_HEIGHT

boundary = {
    "minX": 0.,
    "maxX": 40.,
    "minY": -20.,
    "maxY": 20.,
    "minZ": -0.50,
    "maxZ": 3.50
}

bound_size_x = boundary['maxX'] - boundary['minX']
bound_size_y = boundary['maxY'] - boundary['minY']
bound_size_z = boundary['maxZ'] - boundary['minZ']

boundary_back = {
    "minX": -40.,
    "maxX": 0.,
    "minY": -20.,
    "maxY": 20.,
    "minZ": -0.50,
    "maxZ": 3.50
}

BEV_WIDTH = 416  # across y axis -25m ~ 25m
BEV_HEIGHT = 416  # across x axis 0m ~ 50m
DISCRETIZATION = (boundary["maxX"] - boundary["minX"]) / float(BEV_HEIGHT)

# maximum number of points per voxel
T = 35

# voxel size
vd = 0.1  # z
vh = 0.05  # y
vw = 0.05  # x

# voxel grid
W = math.ceil(bound_size_x / vw)
H = math.ceil(bound_size_y / vh)
D = math.ceil(bound_size_z / vd)

# Following parameters are calculated as an average from KITTI dataset for simplicity
#####################################################################################
Tr_velo_to_cam = np.array([
    [0.430713, -0.410184, -0.00712446, -0.0873391],
    [0.309752, 0.0234157, -0.459227, 0.575476],
    [0.000912053, 7.36036e-05, -1.80851e-05, -0.000329252]
])

#####################################################################################
