"""
Run the learned model to connect to client with ros messages
"""
import pygame
import time
import sys
import fire
import string 
import random
import numpy as np 
sys.path.append('.. ')
sys.path.append('/mnt/intention_net')

# import local file
from joy_teleop import JOY_MAPPING
from policy import Policy
# ros packages
import rospy
from sensor_msgs.msg import Joy, Image, Imu, CompressedImage, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32, Float32, String
from nav_msgs.msg import Odometry
import cv2
from cv_bridge import CvBridge

import torch
from utils.undistort import undistort,FRONT_CAMERA_INFO,RIGHT_CAMERA_INFO,LEFT_CAMERA_INFO
from src.dataset import MultiCamPioneerDataset as Dataset
from skimage.color import rgb2gray

# SCREEN SCALE IS FOR high dpi screen, i.e. 4K screen
SCREEN_SCALE = 1
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768

class Timer(object):
    def __init__(self):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time

class Controller(object):
    tele_twist = Twist()
    def __init__(self, mode, scale_x, scale_z, rate):
        self._mode = mode
        self._scale_x = scale_x
        self._scale_z = scale_z
        self._timer = Timer()
        self._rate = rospy.Rate(rate)
        self._enable_auto_control = False
        self.current_control = None 
        self.trajectory_index = None
        self.bridge = CvBridge()

        # callback data store
        self.image = None
        self.left_img = None 
        self.right_img = None 
        self.depth_img = None 
        self.me1_left = None
        self.me1_right =None
        self.me1_depth = None 
        self.me2_left = None
        self.me2_right = None
        self.me2_depth = None 
        self.me3_left = None
        self.me3_right = None
        self.me3_depth = None  
        self.intention = None
        self.imu = None
        self.imu1 = None
        self.imu2 = None
        self.imu3 = None
        self.odom = None
        self.speed = None
        self.labeled_control = None
        self.key = None
        self.training = False
        self.scan = None
        self.manual_intention = 'forward'
        self.is_manual_intention = False

        # subscribe ros messages
        rospy.Subscriber('/mynteye/left/image_raw/compressed', CompressedImage, self.cb_left_img, queue_size=1, buff_size=2**10)
        rospy.Subscriber('/mynteye/right/image_raw/compressed', CompressedImage, self.cb_right_img, queue_size=1, buff_size=2**10)
        rospy.Subscriber('/mynteye/depth/image_raw/compressed', CompressedImage, self.cb_depth_img, queue_size=1, buff_size=2**10)
        rospy.Subscriber('/mynteye_1/left/image_raw/compressed', CompressedImage, self.cb_me1_left_img, queue_size=1, buff_size=2**10)
        rospy.Subscriber('/mynteye_1/right/image_raw/compressed', CompressedImage, self.cb_me1_right_img, queue_size=1, buff_size=2**10)
        rospy.Subscriber('/mynteye_1/depth/image_raw/compressed', CompressedImage, self.cb_me1_depth_img, queue_size=1, buff_size=2**10)
        rospy.Subscriber('/mynteye_2/left/image_raw/compressed', CompressedImage, self.cb_me2_left_img, queue_size=1, buff_size=2**10)
        rospy.Subscriber('/mynteye_2/right/image_raw/compressed', CompressedImage, self.cb_me2_right_img, queue_size=1, buff_size=2**10)
        rospy.Subscriber('/mynteye_2/depth/image_raw/compressed', CompressedImage, self.cb_me2_depth_img, queue_size=1, buff_size=2**10)
        rospy.Subscriber('/mynteye_3/left/image_raw/compressed', CompressedImage, self.cb_me3_left_img, queue_size=1, buff_size=2**10)
        rospy.Subscriber('/mynteye_3/right/image_raw/compressed', CompressedImage, self.cb_me3_right_img, queue_size=1, buff_size=2**10)
        rospy.Subscriber('/mynteye_3/depth/image_raw/compressed', CompressedImage, self.cb_me3_depth_img, queue_size=1, buff_size=2**10)
        rospy.Subscriber('/scan', LaserScan, self.cb_scan, queue_size=1, buff_size=2**10)
        rospy.Subscriber('/imu', Imu, self.cb_imu, queue_size=1, buff_size=2**10)
        rospy.Subscriber('/imu1', Imu, self.cb_imu1, queue_size=1, buff_size=2**10)
        rospy.Subscriber('/imu2', Imu, self.cb_imu2, queue_size=1, buff_size=2**10)
        rospy.Subscriber('/imu3', Imu, self.cb_imu3, queue_size=1, buff_size=2**10)

        if mode == 'DLM':
            rospy.Subscriber('/test_intention', String, self.cb_dlm_intention, queue_size=1)
        else:
            rospy.Subscriber('/intention_lpe', Image, self.cb_lpe_intention, queue_size=1, buff_size=2**10)
        rospy.Subscriber('/speed', Float32, self.cb_speed, queue_size=1) 
        rospy.Subscriber('/odometry/filtered',Odometry,self.cb_odom,queue_size=1,buff_size=2**10)
        rospy.Subscriber('/joy', Joy, self.cb_joy)
        
        # publish control
        self.control_pub = rospy.Publisher('/RosAria/cmd_vel', Twist, queue_size=1)

        # publishing as training data
        self.pub_intention = rospy.Publisher('/test_intention',String,queue_size=1)
        self.pub_trajectory_index = rospy.Publisher('/train/trajectory_index',String,queue_size=1)
        self.pub_teleop_vel = rospy.Publisher('/train/cmd_vel', Twist, queue_size=1)
        self.pub_left_img = rospy.Publisher('/train/mynteye/left_img/compressed', CompressedImage, queue_size=1)
        self.pub_right_img = rospy.Publisher('/train/mynteye/right_img/compressed',CompressedImage, queue_size=1)
        self.pub_depth_img = rospy.Publisher('/train/mynteye/depth_img/compressed', CompressedImage, queue_size=1)
        self.pub_me1_left_img = rospy.Publisher('/train/mynteye_1/left_img/compressed', CompressedImage, queue_size=1)
        self.pub_me1_right_img = rospy.Publisher('/train/mynteye_1/right_img/compressed',CompressedImage, queue_size=1)
        self.pub_me1_depth_img = rospy.Publisher('/train/mynteye_1/depth_img/compressed',CompressedImage, queue_size=1)
        self.pub_me2_left_img = rospy.Publisher('/train/mynteye_2/left_img/compressed', CompressedImage, queue_size=1)
        self.pub_me2_right_img = rospy.Publisher('/train/mynteye_2/right_img/compressed',CompressedImage, queue_size=1)
        self.pub_me2_depth_img = rospy.Publisher('/train/mynteye_2/depth_img/compressed',CompressedImage, queue_size=1)
        self.pub_me3_left_img = rospy.Publisher('/train/mynteye_3/left_img/compressed', CompressedImage, queue_size=1)
        self.pub_me3_right_img = rospy.Publisher('/train/mynteye_3/right_img/compressed',CompressedImage, queue_size=1)
        self.pub_me3_depth_img = rospy.Publisher('/train/mynteye_3/depth_img/compressed',CompressedImage, queue_size=1)
        self.pub_intention = rospy.Publisher('/train/intention', String, queue_size=1)
        self.pub_imu = rospy.Publisher('/train/imu', Imu, queue_size=1)
        self.pub_imu1 = rospy.Publisher('/train/imu1', Imu, queue_size=1)
        self.pub_imu2 = rospy.Publisher('/train/imu2', Imu, queue_size=1)
        self.pub_imu3 = rospy.Publisher('/train/imu3', Imu, queue_size=1)
        self.pub_odom = rospy.Publisher('/train/odometry/filtered', Odometry, queue_size=1)
        self.pub_scan = rospy.Publisher('/train/scan',LaserScan,queue_size=1)
        

    def cb_scan(self,msg):
        self.scan = msg

    def cb_left_img(self, msg):
        self.left_img = msg

    def cb_right_img(self,msg):
        self.right_img = msg

    def cb_depth_img(self,msg):
        self.depth_img = msg

    def cb_me1_left_img(self,msg):
        self.me1_left = msg
    
    def cb_me1_right_img(self,msg):
        self.me1_right = msg
    
    def cb_me1_depth_img(self,msg):
        self.me1_depth = msg
    
    def cb_me2_left_img(self,msg):
        self.me2_left = msg
    
    def cb_me2_right_img(self,msg):
        self.me2_right = msg
    
    def cb_me2_depth_img(self,msg):
        self.me2_depth = msg

    def cb_me3_left_img(self,msg):
        self.me3_left = msg
    
    def cb_me3_right_img(self,msg):
        self.me3_right = msg

    def cb_me3_depth_img(self,msg):
        self.me3_depth = msg
    
    def cb_dlm_intention(self, msg):
        self.intention = msg.data

    def cb_lpe_intention(self, msg):
        self.intention = cv2.resize(undistort(CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8')), (224, 224))

    def cb_speed(self, msg):
        self.speed = msg.data

    def cb_labeled_control(self, msg):
        self.labeled_control = msg

    def cb_imu(self,msg):
        self.imu = msg
    
    def cb_imu1(self,msg):
        self.imu1 = msg
    
    def cb_imu2(self,msg):
        self.imu2 = msg
    
    def cb_imu3(self,msg):
        self.imu3 = msg

    def cb_odom(self,msg):
        self.odom = msg

    def cb_joy(self, data):
        self.tele_twist.linear.x = self._scale_x * data.axes[JOY_MAPPING['axes']['left_stick_ud']]
        self.tele_twist.angular.z = self._scale_z * data.axes[JOY_MAPPING['axes']['left_stick_lr']]

        # parse control key
        if data.buttons[JOY_MAPPING['buttons']['A']] == 1:
            self._enable_auto_control = True
            print('Auto control')
        if data.buttons[JOY_MAPPING['buttons']['B']] == 1:
            self._enable_auto_control = False
            print('Manual control')
        if data.buttons[JOY_MAPPING['buttons']['back']] == 1:
            self.key = 'q'
        # toggle recording mode and generate trajectory index if start recording
        if data.buttons[JOY_MAPPING['buttons']['start']] == 1:
            self.key = 't'
            print('toggle training mode to: %s'%(not self.training))
        if data.buttons[JOY_MAPPING['buttons']['mode']] == 1:
            self.key = 'i'
            print('toggle intention mode to: %s'%(not self.is_manual_intention))
        #STRAIGHT_FORWARD
        if data.buttons[JOY_MAPPING['buttons']['X']] == 1: 
            self.manual_intention =  'forward'
            print('Intention is manually set to: forward')
        #LEFT_TURN
        if data.buttons[JOY_MAPPING['buttons']['lt']] == 1:
            self.manual_intention = 'left'
            print('Intention is manually set to: left')
        #RIGHT_TURN
        if data.buttons[JOY_MAPPING['buttons']['rt']] == 1:
            self.manual_intention = 'right'
            print('Intention is manually set to: right')

    def _random_string(self,n):
        chars = string.ascii_letters+string.digits
        ret = ''.join(random.choice(chars) for _ in range(n))
        return ret

    def _on_loop(self, policy):
        """
        Logical loop
        """
        self._timer.tick()
        if self.key == 'q':
            #sys.exit(-1)
            self.intention = 'stop'
            self.key = ''
        if self.key == 't':
            self.training = not self.training
            self.key = ''
        if self.key == 'i':
            self.is_manual_intention = not self.is_manual_intention
            self.key = ''
        if self._enable_auto_control:
            if self.is_manual_intention:
                intention = Dataset.INTENTION_MAPPING[self.manual_intention]
            else:
                intention = Dataset.INTENTION_MAPPING[self.intention] # map intention str => int
            if not intention:
                    print('estimate pose + goal....')
            elif intention == 'stop':
                self.tele_twist.linear.x = 0
                self.tele_twist.angular.z = 0
            else:
                print('intention: ',intention)
                # convert ros msg -> cv2
                inp = self.get_data(intention)
                pred_control = np.array(policy(*inp))[0]
                self.tele_twist.linear.x = pred_control[0]*Dataset.SCALE_VEL*0.8
                self.tele_twist.angular.z = pred_control[1]*Dataset.SCALE_STEER*0.8
        
        # publish to /train/* topic to record data (if in training mode)
        if self.training:
            self._publish_as_trn()
        
        # publish control
        self.control_pub.publish(self.tele_twist)
    
    def get_data(self,intention):
        mrgb = cv2.resize(undistort(self.bridge.compressed_imgmsg_to_cv2(self.right_img,desired_encoding='bgr8'),FRONT_CAMERA_INFO),(3,224,224))
        mbnw = rgb2gray(mrgb)
        rbnw = cv2.resize(undistort(self.bridge.compressed_imgmsg_to_cv2(self.me2_left,desired_encoding='bgr8'),RIGHT_CAMERA_INFO),(224,224))
        lbnw = cv2.resize(undistort(self.bridge.compressed_imgmsg_to_cv2(self.me3_right,desired_encoding='bgr8'),LEFT_CAMERA_INFO),(224,224))
        dm = cv2.resize(self.bridge.compressed_imgmsg_to_cv2(self.depth_img,desired_encoding='bgr8'),(224,224))
        dr = cv2.resize(self.bridge.compressed_imgmsg_to_cv2(self.me2_depth,desired_encoding='bgr8'),(224,224))
        dl = cv2.resize(self.bridge.compressed_imgmsg_to_cv2(self.me3_depth,desired_encoding='bgr8'),(224,224))

        #preprocess
        mbnw = torch.tensor(mbnw).expand(1,1,224,224).float()/255.0
        rbnw = torch.tensor(rbnw).expand(1,1,224,224).float()/255.0
        lbnw = torch.tensor(lbnw).expand(1,1,224,224).float()/255.0
        dm = torch.tensor(dm).expand(1,1,224,224).float()/255.0
        dr = torch.tensor(dr).expand(1,1,224,224).float()/255.0
        dl = torch.tensor(dl).expand(1,1,224,224).float()/255.0

        intention = torch.tensor([intention]).long()

        return [intention,dl,dm,dr,lbnw,mbnw,rbnw]

    def _publish_as_trn(self):
        if self.odom:
            self.pub_trajectory_index.publish(self.trajectory_index)
            self.pub_left_img.publish(self.left_img)
            self.pub_right_img.publish(self.right_img)
            self.pub_depth_img.publish(self.depth_img)
            self.pub_me1_left_img.publish(self.me1_left)
            self.pub_me1_right_img.publish(self.me1_right)
            self.pub_me1_depth_img.publish(self.me1_depth)
            self.pub_me2_left_img.publish(self.me2_left)
            self.pub_me2_right_img.publish(self.me2_right)
            self.pub_me2_depth_img.publish(self.me2_depth)
            self.pub_me3_left_img.publish(self.me3_left)
            self.pub_me3_right_img.publish(self.me3_right)
            self.pub_me3_depth_img.publish(self.me3_depth)
            self.pub_teleop_vel.publish(self.tele_twist)
            self.pub_intention.publish(self.intention)
            self.pub_imu.publish(self.imu)
            self.pub_imu1.publish(self.imu1)
            self.pub_imu2.publish(self.imu2)
            self.pub_imu3.publish(self.imu3)
            self.pub_odom.publish(self.odom)
            self.pub_scan.publish(self.scan)

    def execute(self, policy):
        while True:
            self._on_loop(policy)
            self._rate.sleep()

# wrapper for fire to get command arguments
def run_wrapper(mode='DLM', input_frame='NORMAL', model_path=None, num_intentions=3, scale_x=1, scale_z=1, rate=10):
    rospy.init_node("joy_controller")
    controller = Controller(mode, scale_x, scale_z, rate)
    if model_path == None:
        policy = None
    else:    
        policy = torch.load(model_path)
    controller.execute(policy)

def main():
    fire.Fire(run_wrapper)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print ('\nCancelled by user! Bye Bye!')
