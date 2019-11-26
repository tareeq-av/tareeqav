TareeqAV
======

Simulation
------

The intent of the  TareeqAV platform is to step out of the simulator and into the hands-on practice of both the hardware and software challenges and systems engineering of building a self-driving car.

Rather than use a simulator, we use the [KITTI Raw Dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) as our sandbox.  The benefits of doing this is that the KITTI Raw Data gives us many different complete driving scenes that range from 30 seconds to 1.5 minutes of urban city driving, with data from:

1. 4 Front Facing Synchronized Cameras
2. Velodyne LiDAR Data
3. GPS/IMU Data

We test our code against the data of these sensors and annotate the main camera (camera 02) with the output of the __perception__ and __planning__ modules.

Physical Platform
------

**Currently Under Development**

The documentation for a physical platform will allow anyone to run the TareeqAV platform on easy to obtain hardware, such as Nvidia TX2/Xavier.


Stage 1 Implementation
------

The Stage 1 implementation __stitches__ together the output of various deep learning models from top ranking research papers in computer vision conferences.

We use the model and the code submitted with the research paper as is, and keep the commit history of the original authors.  We change only python import statements to match the overall platform.

Using these models as single taks networks comes with high inefficiency, however we accept this since *Stage 1* intends to ease the engineer into the complexity of building a complete self-driving car stack from the groud up.

Some of these research papers use Tensorflow, while others use Pytorch.  We make no effort to port any of the code, and only ensure that it can run on our platform.

Usnig these networks we implement the following self-driving modules:

1. Perception
2. Planning
3. Control

###### Running Stage 1

This repository comes with a single sample from two of the four cameras, the LiDAR data and the calibration files from one of the KITTI Raw Dataset scenes.  We do this to allow engineers to preview what the system can do and provide them a place to begin exploring.

The __data_samples__ directory in the main __python__ directory contains all the files necessary and the __pipeline.py__ module in the __stage1__ contains path refernces and is ready to run without any modifications.

Please refer to the [README](./perception/README.md) on how to install the software dependenices of the perception module.

Once all dependenices are installed, issuing the following command should open a single annotated image with 3D cuboids, detected lane lines, and driveable space.

```
cd <repo-root>/python/stage1/
source venv/bin/activate # refer to README in perception
python3 demp_pipeline.py
```

Perception
------


* For details on how to install and run the module locally, please refer to the [README](./perception/README.md) of the perception directory.

We use the state-of-the-art in computer vision research based on the benchmarks for [KITTI 3D Object](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d), and the [TuSimple Lane Detection]https://github.com/TuSimple/tusimple-benchmark).

The perception stack comprises the following tasks:

1. Traffic Sign Detection and Recognition
2. 3D Object Detection and Localization
3. Lane Detection
4. Drivable Space Estimation
5. Distance To Impact

3D Object Detection
------

We use the widely adopted PointNet++ deep networks as impelemnted by [PointRCNN](https://arxiv.org/pdf/1812.04244.pdf).


Lane Detection
------

###### Installation
This software has been tested on ubuntu 18.04(x64), python3.6, cuda-10.2, cudnn-7.0 with a GTX-1080ti GPU. 
To install this software you need tensorflow 1.14.2 

```
pip3 install -r requirements.txt
```
###### Inference

Step1: Download the model files from [here](https://www.dropbox.com/sh/tnsf0lw6psszvy4/AAA81r53jpUI3wLsRW6TiPCya?dl=0)
Step2: Store them in the folder ./model/tusimple_lanenet_vgg/
Step3: Download the data and store it in any folder for eg ./data/
Your final folder structure should look something like:-

--perception
	--lanenet
		--config
		--data
		--lanenet_model
		--model
			--tusimple_lanenet_vgg
				--checkpoint
				--tusimple_lanenet_vgg.ckpt.data-00000-of-00001
				--tusimple_lanenet_vgg.ckpt.index
				--tusimple_lanenet_vgg.ckpt.meta
		--semantic_segmentation_zoo
		--tools

Step4: python predict.py path/to/the/raw/video


###### Training your model

You may call the following script to train your own model

```
python tools/train_lanenet.py 
--net vgg 
--dataset_dir ./data/training_data_example
-m 0
```
You can also continue the training process from the snapshot by
```
python tools/train_lanenet.py 
--net vgg 
--dataset_dir data/training_data_example/ 
--weights_path path/to/your/last/checkpoint
-m 0
```
	


