"""
load the learned model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import scipy.misc

from keras.utils import to_categorical
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# intention net package
import sys
sys.path.append('/mnt/intention_net')
from net import IntentionNet
from dataset import preprocess_input
from dataset import PioneerDataset as Dataset
import matplotlib.pyplot as plt

class Policy(object):
    def __init__(self, mode, input_frame, num_control, path, num_intentions, gpu_fraction=0.75, vis=False):
        # set keras session
        config_gpu = tf.ConfigProto()
        config_gpu.gpu_options.allow_growth = True
        config_gpu.gpu_options.per_process_gpu_memory_fraction=gpu_fraction
        KTF.set_session(tf.Session(config=config_gpu))
        self.model = None
        self.mode = mode
        self.input_frame = input_frame
        self.num_control = num_control
        self.num_intentions = num_intentions
        self.path = path
        self.load_model()
        self.vis = vis
        if self.vis:
            self.count = 0
            self.fig = plt.figure()

    def load_model(self):
        model = IntentionNet(self.mode, self.input_frame, self.num_control, self.num_intentions)
        # load checkpoint
        fn = osp.join(self.path, self.input_frame + '_' + self.mode+'_latest_model.h5')
        model.load_weights(fn)
        print ("=> loaded checkpoint '{}'".format(fn))
        self.model = model

    def predict_control(self, image, intention, speed, segmented=False):
        if self.input_frame == 'MULTI':
            rgb = [np.expand_dims(preprocess_input(im), axis=0) for im in image]
        else:
            if segmented:
                rgb = [np.expand_dims(image, axis=0)]
            else:
                rgb = [np.expand_dims(preprocess_input(image), axis=0)]

        if self.mode == 'DLM':
            i_intention = to_categorical([intention], num_classes=self.num_intentions)
        else:
            i_intention = np.expand_dims(preprocess_input(intention), axis=0)

        if self.input_frame == 'NORMAL':
            pred_control = self.model.predict(rgb + [i_intention] + [speed])
            # pred_control = self.model.predict([rgb,i_intention])
        elif self.input_frame == 'MULTI':
            pred_control = self.model.predict(rgb+[i_intention])

        if self.vis:
            self.fig.clf()
            self.count += 1
            if self.mode == 'DLM':
                text = f'speed {speed}\nsteer {2*pred_control[0][0]*np.pi}\nacc {pred_control[0][1]*0.8}\nintention {intention}'
                if self.input_frame == 'MULTI':
                    ax0 = self.fig.add_subplot(131)
                    ax1 = self.fig.add_subplot(132)
                    ax2 = self.fig.add_subplot(133)
                    ax0.imshow(image[0])
                    ax1.imshow(image[1])
                    ax2.imshow(image[2])
                else:
                    ax0 = self.fig.add_subplot(111)
                    ax0.imshow(image)

            else:
                text = f'speed {speed}\nsteer {2*pred_control[0][0]*np.pi}\nacc {pred_control[0][1]*0.8}'
                if self.input_frame == 'MULTI':
                    ax0 = self.fig.add_subplot(141)
                    ax1 = self.fig.add_subplot(142)
                    ax2 = self.fig.add_subplot(143)
                    ax3 = self.fig.add_subplot(144)
                    ax0.imshow(image[0])
                    ax1.imshow(image[1])
                    ax2.imshow(image[2])
                    ax3.imshow(intention)
                else:
                    ax0 = self.fig.add_subplot(121)
                    ax1 = self.fig.add_subplot(122)
                    ax1.imshow(intention)

            self.fig.suptitle(text)
            self.fig.savefig(f'debug/step_{self.count:08d}.png')

        return pred_control
