# Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds

[![python-image]][python-url]
[![pytorch-image]][pytorch-url]

---

## Features
- [x] Super fast and accurate 3D object detection based on LiDAR
- [x] Fast training, fast inference
- [x] An Anchor-free approach
- [x] No Non-Max-Suppression
- [x] Support [distributed data parallel training](https://github.com/pytorch/examples/tree/master/distributed/ddp)
- [x] Release pre-trained models 

**The technical details are described [here](./Technical_details.md)**

**Update 2020.09.06**: Add `ROS` node. The great work has been done by @AhmedARadwan. The instructions for using the ROS code could be found [here](https://github.com/maudzung/Super-Fast-Accurate-3D-Object-Detection/blob/master/ros/src/super_fast_object_detection/README.md)

## Demonstration (on a single GTX 1080Ti)

[![demo](http://img.youtube.com/vi/FI8mJIXkgX4/0.jpg)](http://www.youtube.com/watch?v=FI8mJIXkgX4)


**[Youtube link](https://youtu.be/FI8mJIXkgX4)**

## 2. Getting Started
### 2.1. Requirement

```shell script
cd ~/catkin_ws/src
git clone https://github.com/hyungikim4/SFA3D.git
cd SFA3D/
pip install .

cd ~/catkin_ws && catkin_make
```

### 2.2. Data Preparation
Download the 3D KITTI detection dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

The downloaded data includes:

- Velodyne point clouds _**(29 GB)**_
- Training labels of object data set _**(5 MB)**_
- Camera calibration matrices of object data set _**(16 MB)**_
- **Left color images** of object data set _**(12 GB)**_ (For visualization purpose only)


Please make sure that you construct the source code & dataset directories structure as below.

### 2.3. How to run

#### 2.3.1. Visualize the dataset 

To visualize 3D point clouds with 3D boxes, let's execute:

```shell script
cd sfa/data_process/
python kitti_dataset.py
```


#### 2.3.2. Inference

The pre-trained model was pushed to this repo.

```
python test.py --gpu_idx 0 --peak_thresh 0.2
```

#### 2.3.3. Making demonstration

```
python demo_2_sides.py --gpu_idx 0 --peak_thresh 0.2
```

The data for the demonstration will be automatically downloaded by executing the above command.


#### 2.3.4. Training

##### 2.3.4.1. Single machine, single gpu

```shell script
python train.py --gpu_idx 0
```

##### 2.3.4.2. Distributed Data Parallel Training
- **Single machine (node), multiple GPUs**

```
python train.py --multiprocessing-distributed --world-size 1 --rank 0 --batch_size 64 --num_workers 8
```

- **Two machines (two nodes), multiple GPUs**

   - _**First machine**_
    ```
    python train.py --dist-url 'tcp://IP_OF_NODE1:FREEPORT' --multiprocessing-distributed --world-size 2 --rank 0 --batch_size 64 --num_workers 8
    ```

   - _**Second machine**_
    ```
    python train.py --dist-url 'tcp://IP_OF_NODE2:FREEPORT' --multiprocessing-distributed --world-size 2 --rank 1 --batch_size 64 --num_workers 8
    ```

### 2.4. Training and Testing on your own dataset

Since it is difficult to create a large amount of custom data, it is recommended to perform fine tuning with the custom dataset after training according to the desired config with the KITTI dataset.

#### 2.4.1. Prepare your Dataset

##### 2.4.1.1. Setting config

```shell script
cd ros/src/super_fast_object_detection/src/
```
Modify `veloster_config_2sides.py` to fit your dataset. 

|Parameter| Description|
----------|--------
|`boundary`|Front lidar x,y,z range. **Range of z is dependent on the height of Lidar sensor.**|
|`boundary_back`|Back lidar x,y,z range.|
|`BEV_WIDTH`|The width of BEV RGB map.|
|`BEV_HEIGHT`|The height of BEV RGB map.|
|`Tr_velo_to_cam`|Transform matrix(3x4) from Lidar coordinate to Image pixel coordinate.|

##### 2.4.1.2. Make training dataset

Save the Lidar raw pointcloud, front image, and BEV RGB map at 1 Hz from the rosbag in which the Lidar data and front image data are logged.

For easily labeling, BEV binary map of ground filtered pointcloud and BEV RGB map are saved.

```
roslaunch points_preprocessor_usi groundfilter.launch
python make_training_data_for_labeling_2sides_without_map.py
```

```plain
└── save_path
       ├── bev          <-- BEV RGB map image
       ├── front_image  <-- front RGB image
       ├── lidar        <-- Lidar raw data (.npy)
       └── label        <-- label data (**TO DO**)
```

![input](/images/Label_images.png?raw=true)

##### 2.4.1.3. Auto-labeling

Auto-labeling using the trained model

`python3 inference_label_2sides.py`

|Parameter| Type| Description|
----------|-----|--------
|`self.conf_thres`|*double* |Confidence threshold. Only objects with confidence higher than `self.conf_thres` are stored in the label text.|
|`configs.pretrained_path`|*string*|Pretrained model path|
|`data_root`|*string*|Root path of your dataset.|
|`have_train_txt`|*bool*|If it is true, inference only for data in `train.txt`|

##### 2.4.1.4. labeling

Follow this guideline.

https://github.com/zexihan/labelImg-kitti

##### 2.4.1.5. Make ImageSets

```shell script
cd ~/catkin_ws/src/SFA3D/sfa/data_process
python write_train_txt.py
```

##### 2.4.1.6. Visualize the dataset

Visualize 3D point clouds with 3D boxes in Rviz. 

```shell script
cd ~/catkin_ws/src/SFA3D/sfa/data_process
python3 veloster_2sides_dataset.py
```

![visualization](/images/Visualize_the_dataset.png?raw=true)

#### 2.4.2. Training

##### 2.4.2.1. Setting config

```shell script
cd ~/catkin_ws/src/SFA3D
cp ros/src/super_fast_object_detection/src/veloster_config_2sides.py sfa/config/
```

##### 2.4.2.1. Setting training config

|Parameter| Type| Description|
----------|-----|--------
|`saved_fn`|*string* |Path where model will be saved|
|`pretrained_path`|*string*|Pretrained model path|
|`configs.input_size`|*int*|The image size of BEV RGB map.|
|`configs.hm_size`|*int*|`configs.input_size / configs.down_ratio`|
|`configs.num_classes`|*int*|The number of classes|

##### 2.4.2.1. Training

```shell script
cd ~/catkin_ws/src/SFA3D/sfa
python3 train.py
```

#### 2.4.3. Inference with ROS

```shell script
source ~/catkin_ws/devel/setup.bash
rosrun super_fast_object_detection rosInference_2sides.py
```

#### Tensorboard

- To track the training progress, go to the `logs/` folder and 

```shell script
cd logs/<saved_fn>/tensorboard/
tensorboard --logdir=./
```

- Then go to [http://localhost:6006/](http://localhost:6006/)


## Citation

```bash
@misc{Super-Fast-Accurate-3D-Object-Detection-PyTorch,
  author =       {Nguyen Mau Dung},
  title =        {{Super-Fast-Accurate-3D-Object-Detection-PyTorch}},
  howpublished = {\url{https://github.com/maudzung/Super-Fast-Accurate-3D-Object-Detection}},
  year =         {2020}
}
```

## References

[1] CenterNet: [Objects as Points paper](https://arxiv.org/abs/1904.07850), [PyTorch Implementation](https://github.com/xingyizhou/CenterNet) <br>
[2] RTM3D: [PyTorch Implementation](https://github.com/maudzung/RTM3D) <br>
[3] Libra_R-CNN: [PyTorch Implementation](https://github.com/OceanPang/Libra_R-CNN)

_The YOLO-based models with the same BEV maps input:_ <br>
[4] Complex-YOLO: [v4](https://github.com/maudzung/Complex-YOLOv4-Pytorch), [v3](https://github.com/ghimiredhikura/Complex-YOLOv3), [v2](https://github.com/AI-liu/Complex-YOLO)

*3D LiDAR Point pre-processing:* <br>
[5] VoxelNet: [PyTorch Implementation](https://github.com/skyhehe123/VoxelNet-pytorch)

## Folder structure

```
${ROOT}
└── checkpoints/
    ├── fpn_resnet_18/    
        ├── fpn_resnet_18_epoch_300.pth
└── dataset/    
    └── kitti/
        ├──ImageSets/
        │   ├── test.txt
        │   ├── train.txt
        │   └── val.txt
        ├── training/
        │   ├── image_2/ (left color camera)
        │   ├── calib/
        │   ├── label_2/
        │   └── velodyne/
        └── testing/  
        │   ├── image_2/ (left color camera)
        │   ├── calib/
        │   └── velodyne/
        └── classes_names.txt
└── sfa/
    ├── config/
    │   ├── train_config.py
    │   └── kitti_config.py
    ├── data_process/
    │   ├── kitti_dataloader.py
    │   ├── kitti_dataset.py
    │   └── kitti_data_utils.py
    ├── models/
    │   ├── fpn_resnet.py
    │   ├── resnet.py
    │   └── model_utils.py
    └── utils/
    │   ├── demo_utils.py
    │   ├── evaluation_utils.py
    │   ├── logger.py
    │   ├── misc.py
    │   ├── torch_utils.py
    │   ├── train_utils.py
    │   └── visualization_utils.py
    ├── demo_2_sides.py
    ├── demo_front.py
    ├── test.py
    └── train.py
└── ros/
├── README.md 
└── requirements.txt
```



[python-image]: https://img.shields.io/badge/Python-3.6-ff69b4.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.5-2BAF2B.svg
[pytorch-url]: https://pytorch.org/
