"""
Run the learned model to connect to simulator
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import scipy.misc

from carla.agent import Agent
from carla.carla_server_pb2 import Control
from easydict import EasyDict as edict

from keras.utils import to_categorical
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# intention net package
from intention_net.net import IntentionNet
from intention_net.dataset import preprocess_input, intention_mapping

class IntentionNetAgent(Agent):
    STEER = 0
    GAS = 1
    NUM_INTENTIONS = 5

    def __init__(self, city_name, mode, num_control, path=None, image_cut=[115, 510], gpu_fraction=0.75):
        Agent.__init__(self)
         # set keras session
        config_gpu = tf.ConfigProto()
        config_gpu.gpu_options.allow_growth = True
        config_gpu.gpu_options.per_process_gpu_memory_fraction=gpu_fraction
        KTF.set_session(tf.Session(config=config_gpu))
        self.model = None
        self.mode = mode
        self.num_control = num_control
        self.path = path
        self._image_cut = image_cut
        self.init()

    def init(self):
        model = IntentionNet(self.mode, self.num_control, self.NUM_INTENTIONS)
        # load checkpoint
        fn = osp.join(self.path, self.mode+'_best_model.h5')
        model.load_weights(fn)
        print ("=> loaded checkpoint '{}'".format(fn))
        self.model = model

    def run_step(self, measurements, sensor_data, directions, target):
        rgb = sensor_data['CameraRGB'].data
        # publish data need to cut
        rgb = rgb[self._image_cut[0]:self._image_cut[1], :]
        rgb = scipy.misc.imresize(rgb, (224, 224))
        rgb = np.expand_dims(preprocess_input(rgb), axis=0)

        if self.mode == 'DLM':
            intention = to_categorical([intention_mapping[directions]], num_classes=self.NUM_INTENTIONS)
        else:
            intention = np.expand_dims(preprocess_input(directions), axis=0)

        speed = np.array([measurements.player_measurements.forward_speed])

        pred_control = self.model.predict([rgb, intention, speed])
        control = Control()
        control.steer = pred_control[0][self.STEER]
        control.throttle = pred_control[0][self.GAS]

        if control.throttle < 0.0:
            control.brake = -control.throttle
            control.throttle = 0.0

        control.hand_brake = 0
        control.reverse = 0
        return control, None
