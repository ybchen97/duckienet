"""
simulate rostopics from dataset for testing
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import fire
import numpy as np
from tqdm import tqdm

# ros packages
import rospy
from std_msgs.msg import Float64, Int32
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge

# import local file
from policy import Policy
# include quantitative metrics
from intention_net.statistics import SmoothStatistics as Stats

NUM_INTENTIONS = 3

def plot_wrapper(dataset, data_dir, mode, input_frame, model_dir, num_intentions=NUM_INTENTIONS):
    import matplotlib.pyplot as plt
    if dataset == 'CARLA':
        from intention_net.dataset import CarlaImageDataset as Dataset
        print ('=> use CARLA published data')
    elif dataset == 'CARLA_SIM':
        from intention_net.dataset import CarlaSimDataset as Dataset
        print ('=> use CARLA self-collected data')
    else:
        from intention_net.dataset import HuaWeiFinalDataset as Dataset
        print ('=> use HUAWEI data')

    sim_loader = Dataset(data_dir, 1, num_intentions, mode, preprocess=False, input_frame=input_frame)
    policy = Policy(mode, input_frame, 2, model_dir, num_intentions)
    ground_truth = []
    pred_control = []
    speeds = []
    # parse statistics
    stat = Stats(input_frame, mode)
    fig = plt.figure()
    for step, (x, y) in enumerate(tqdm(sim_loader)):
        if (step == len(sim_loader)):
            break

        if input_frame != "MULTI":
            img = x[0][0].astype(np.uint8)
            intention = x[1][0]
            if mode == 'DLM':
                intention = np.argmax(intention)
            else:
                intention = intention.astype(np.uint8)
            speed = x[2][0, 0]
            control = y[0]
            pred = policy.predict_control(img, intention, speed)[0]
        else:
            img_l = x[0][0].astype(np.uint8)
            img_m = x[1][0].astype(np.uint8)
            img_r = x[2][0].astype(np.uint8)

            intention = x[3][0]
            if mode == 'DLM':
                intention = np.argmax(intention)
            else:
                intention = intention.astype(np.uint8)
            speed = x[4][0, 0]
            control = y[0]
            pred = policy.predict_control([img_l, img_m, img_r], intention, speed)[0]

        # scale back
        control[0] *= Dataset.SCALE_STEER
        control[1] *= Dataset.SCALE_ACC
        pred[0] *= Dataset.SCALE_STEER
        pred[1] *= Dataset.SCALE_ACC
        stat.include([speed, pred[0]])
        # add data for plot
        ground_truth.append(control)
        pred_control.append(pred)
        speeds.append(speed)
    # change to numpy array
    ground_truth = np.array(ground_truth)
    pred_control = np.array(pred_control)
    speeds = np.array(speeds)

    stat.log()
    print (stat.str())
    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    x = np.arange(len(sim_loader))
    ax1.plot(x, ground_truth[:,0], 'k', lw=2)
    ax1.plot(x, pred_control[:,0], 'r', lw=2)
    ax2.plot(x, ground_truth[:,1], 'k', lw=2)
    ax2.plot(x, pred_control[:,1], 'r', lw=2)
    ax1.plot(x, speeds, 'g', lw=4)
    ax2.plot(x, speeds, 'g', lw=4)
    fig.suptitle(mode, fontsize="x-large")
    ax1.set_xlabel('Time')
    ax2.set_xlabel('Time')
    ax1.set_ylabel('Steer')
    ax2.set_ylabel('Acceleration')
    plt.show()

def img_to_msg(img, encoding='rgb8'):
    return CvBridge().cv2_to_imgmsg(img, encoding=encoding)

def main_wrapper(dataset, data_dir, mode, input_frame, num_intentions=NUM_INTENTIONS):
    if dataset == 'CARLA':
        from intention_net.dataset import CarlaImageDataset as Dataset
        print ('=> use CARLA published data')
    elif dataset == 'CARLA_SIM':
        from intention_net.dataset import CarlaSimDataset as Dataset
        print ('=> use CARLA self-collected data')
    else:
        from intention_net.dataset import HuaWeiFinalDataset as Dataset
        print ('=> use HUAWEI data')

    # create rosnode
    rospy.init_node('simulate')
    # only for debug so make it slow
    rate = rospy.Rate(28)
    rgb_pub = rospy.Publisher('/image', Image, queue_size=1)
    f_pub = rospy.Publisher('/front_96_image', Image, queue_size=1)
    l_pub = rospy.Publisher('/fl_96_image', Image, queue_size=1)
    r_pub = rospy.Publisher('/fr_96_image', Image, queue_size=1)
    speed_pub = rospy.Publisher('/speed', Imu, queue_size=1)
    control_pub = rospy.Publisher('/control', Imu, queue_size=1)
    if mode == 'DLM':
        intention_pub = rospy.Publisher('/intention_dlm', Imu, queue_size=1)
    else:
        intention_pub = rospy.Publisher('/intention_lpe', Image, queue_size=1)

    sim_loader = Dataset(data_dir, 1, num_intentions, mode, preprocess=False, input_frame=input_frame)
    for step, (x, y) in enumerate(tqdm(sim_loader)):
        if input_frame != 'MULTI':
            img = x[0][0].astype(np.uint8)
            img = img_to_msg(img)
            # publish topics
            rgb_pub.publish(img)

            offset = 0
        else:
            img_l = img_to_msg(x[0][0].astype(np.uint8))
            img_f = img_to_msg(x[1][0].astype(np.uint8))
            img_r = img_to_msg(x[2][0].astype(np.uint8))
            l_pub.publish(img_l)
            rgb_pub.publish(img_f)
            r_pub.publish(img_r)
            offset = 2

        intention = x[offset+1][0]
        if mode == 'DLM':
            intention = np.argmax(intention)
        else:
            intention = intention.astype(np.uint8)
            intention = img_to_msg(intention)
        speed = x[offset+2][0, 0]
        control = y[0]
        ci = Imu()
        ci.linear_acceleration.x = control[1]*Dataset.SCALE_ACC
        ci.angular_velocity.z = control[0]*Dataset.SCALE_STEER
        if mode == 'DLM':
            i = Imu()
            i.linear_acceleration.x = intention
            intention_pub.publish(i)
        else:
            intention_pub.publish(intention)
        i = Imu()
        i.linear_acceleration.x = speed
        speed_pub.publish(i)
        control_pub.publish(ci)
        rate.sleep()

def main():
    fire.Fire({
        'plot': plot_wrapper,
        'ros': main_wrapper,
    })

if __name__ == '__main__':
    main()
