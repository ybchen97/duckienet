# Intention-Net Journal Code

### Installation

Intention-Net requires [Keras](https://keras.io/) v2.2+ and [Tensorflow](https://www.tensorflow.org/) v1.8+ to run.
Python 3.5+
Ros Kinetic

Install virtual environment, miniconda3 is recommended.
Install keras and tensorflow.

```sh
$ conda create -n IntentionNet
$ source activate IntentionNet
$ pip install keras
$ pip install tensorflow-gpu
```

Install requirements

```sh
$ pip install fire
$ pip install tqdm
$ pip install toolz
$ pip install rospkg
$ pip install opencv-python
```

### Ros comflicts with python3

Ros is comflict with python3, here we compile the used ros package from source. 
[vision_opencv](http://wiki.ros.org/vision_opencv): Create a new folder at home folder (or any place you like) named catkin_cv. 

```sh
cd ~/catkin_cv
mkdir src && cd src
git clone https://github.com/ros-perception/vision_opencv.git
# resolve link error of python 3 before compile.
cd /usr/lib/x86_64-linux-gnu/
sudo ln -s libboost_python-py3x.so libboost_python3.so
# compile
cd ~/catkin_cv
catkin_make
# add to the PYTHONPATH
source ~/catkin_cv/devel/setup.bash
# resolve cv2 comflict. The dynamic link of cv2 is broken in python3 together with ros.
echo $PYTHONPATH
# delete ros related path
python
>> import cv2
>> print (cv2.__file__) # get the cv2 path in python3
# add this path directory to the front of PYTHONPATH
# add back rospath
source ~/catkin_cv/devel/setup.bash
source /opt/ros/kinetic/setup.bash
export PYTHONPATH=this_path_dir:$PYTHONPATH
# install the intetion net package
cd "the setup.py folder"
pip install .
```
