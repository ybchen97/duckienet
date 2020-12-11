"""
Generate data from pioneer rosbag
"""
import cv2
import fire
import rosbag
import glob
import shutil
import os
import csv
import os.path as osp
import numpy as np
from tqdm import tqdm
from munch import Munch
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, Imu, CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32, Float32
from toolz import partition_all
from itertools import chain

# import local
import sys
sys.path.append('..')
sys.path.append('/mnt/intention')
from threadedgenerator import ThreadedGenerator
from utils.undistort import undistort,FRONT_CAMERA_INFO,LEFT_CAMERA_INFO,RIGHT_CAMERA_INFO

SENSORS = ['mynt_eye', 'web_cam']

MYNT_EYE = {
    'RGBS' : ['/train/mynteye/right_img/compressed',\
        '/train/mynteye_2/left_img/compressed',\
        '/train/mynteye_3/right_img/compressed'],
    'DEPTHS' : ['/train/mynteye/depth_img/compressed','/train/mynteye_2/depth_img/compressed','/train/mynteye_3/depth_img/compressed'],
}

SENSOR_TOPIC = {
    SENSORS[0] : MYNT_EYE,
}

INTENTION = '/train/intention'
CONTROL = '/train/cmd_vel'

IMG_TOPICS = []
TOPICS = []
TOPICS_IDX = {}

# CHUNK_SIZE for parallel parsing
CHUNK_SIZE = 1

def imgmsg_to_cv2(msg,undistort_param):
    im = CvBridge().compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
    if undistort_param:
        im = undistort(im,undistort_param)
    return cv2.resize(im, (224, 224))

def parse_bag(bagfn, intention_type):
    print (f'processing {bagfn} now')
    bag = rosbag.Bag(bagfn)
    imgs = [None]*len(IMG_TOPICS)
    vel = None
    steer = None
    start = False
    dlm = ''

    print ('img topics', IMG_TOPICS)
    print ('topics', TOPICS)

    def gen(frame):
        return Munch(frame=frame, imgs=imgs, vel=vel, steer=steer, dlm=dlm)

    msg_cnt = [bag.get_message_count(topic) for topic in IMG_TOPICS]
    least_img_idx = msg_cnt.index(min(msg_cnt))

    for topic, msg, frame in bag.read_messages(topics=TOPICS):
        if topic in IMG_TOPICS:
            idx = TOPICS_IDX[topic]
            if topic == '/train/mynteye/right_img/compressed':
                imgs[idx] = imgmsg_to_cv2(msg,FRONT_CAMERA_INFO)
            elif topic == '/train/mynteye_2/left_img/compressed':
                imgs[idx] = imgmsg_to_cv2(msg,RIGHT_CAMERA_INFO)
            elif topic == '/train/mynteye_3/right_img/compressed':
                imgs[idx] = imgmsg_to_cv2(msg,LEFT_CAMERA_INFO)
            else: #depth
                imgs[idx] = imgmsg_to_cv2(msg,None)
            if idx == least_img_idx:
                if start:
                    yield gen(frame)
                else:
                    if sum([1 for it in imgs if it is None]) == 0:
                        start = True
        elif topic == CONTROL:
            vel = msg.linear.x
            steer = msg.angular.z
        elif intention_type == 'dlm' and topic == INTENTION:
            dlm = msg.data

    bag.close()

def main_wrapper(data_dir='/data', sensor='mynt_eye', intention_type='dlm'):
    global IMG_TOPICS
    global TOPICS
    global TOPICS_IDX
    assert (sensor in SENSORS), f'Must be valid sensor in {SENSORS}'
    RGBS = SENSOR_TOPIC[sensor]['RGBS']
    DEPTHS = SENSOR_TOPIC[sensor]['DEPTHS']
    bagfns = glob.glob(data_dir + '/*.bag')
    print ('bags:', bagfns)
    bags = chain(*[parse_bag(bagfn, intention_type) for bagfn in bagfns])
    it = ThreadedGenerator(bags, queue_maxsize=6500)
    # make dirs for images
    gendir = 'data'
    if os.path.exists(osp.join(data_dir, gendir)) and os.path.isdir(osp.join(data_dir, gendir)):
        shutil.rmtree(osp.join(data_dir, gendir))
    os.mkdir(osp.join(data_dir, gendir))
    with open(osp.join(data_dir, gendir,'README.txt'),'w+') as f:
        f.write('THIS DATA IS PARSED TO SERVE PYTORCH MODEL')

    topic_save_path = []
    for idx, rgb_topic in enumerate(RGBS):
        fn = osp.join(data_dir, gendir, f'rgb_{idx}')
        os.mkdir(fn)
        TOPICS.append(rgb_topic)
        TOPICS_IDX[rgb_topic] = len(TOPICS) - 1
        topic_save_path.append(fn)
    for idx, depth_topic in enumerate(DEPTHS):
        fn = osp.join(data_dir, gendir, f'depth_{idx}')
        os.mkdir(fn)
        TOPICS.append(depth_topic)
        TOPICS_IDX[depth_topic] = len(TOPICS) - 1
        topic_save_path.append(fn)
    if intention_type == 'lpe':
        fn = osp.join(data_dir, gendir, 'intention_img')
        os.mkdir(fn)
        TOPICS.append(INTENTION)
        TOPICS_IDX[INTENTION] = len(TOPICS) - 1
        topic_save_path.append(fn)
        IMG_TOPICS = TOPICS[:]
    else:
        IMG_TOPICS = TOPICS[:]
        TOPICS.append(INTENTION)
    TOPICS.append(CONTROL)

    f = open(osp.join(data_dir, gendir, 'label.txt'), 'w')
    labelwriter = csv.writer(f, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    labelwriter.writerow(['frame', 'intention_type', 'current_velocity', 'steering_wheel_angle', 'dlm'])
    for chunk in partition_all(CHUNK_SIZE, tqdm(it)):
        for c in chunk:
            for idx, fn in enumerate(topic_save_path):
                cv2.imwrite(osp.join(fn, f'{c.frame}.jpg'), c.imgs[idx])
            labelwriter.writerow([c.frame, intention_type, c.vel, c.steer, c.dlm])

if __name__ == '__main__':
    fire.Fire(main_wrapper)
