TareeqAV Stage 1
======

Perception
------

The following instructions have ( so far only) been tested against the following:

1. Ubuntu 18.04 on X86_64
2. Python 3.7

###### Installation

```
# install python3's virtual env module
sudo apt-get -y update
sudo apt-get -y install python3-venv

# install OpenCV dependenices
# NOTE: we use OpenCV for annotation of the module's output and not for any processing

# clone repo and setup virtual env
git clone https://github.com/tareeq-av/tareeqav.git
cd tareeqav/python/stage1
python3 -m venv venv
source venv/bin/activate

# install requirements
pip3 install -r perception_requirements.txt

# install Pytorch C++ extensions for PointNet++
cd perception/pointrcnn/pointnet2/
python3 setup.py install

cd - # should take you back to <repo-root>/python/stage1/
cd perception/pointrcnn/lib/utils/roipool3d/
python3 setup.py install

cd - # should take you back to <repo-root>/python/stage1/
cd perception/pointrcnn/lib/utils/iou3d/
python3 setup.py install
```

