from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app as absl_app
from absl import flags
import numpy as np
import os
import sys
from tqdm import tqdm
from time import time

# keras import
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import TensorBoard
from keras.utils.training_utils import multi_gpu_model
from keras_radam import RAdam

from config import *
from net import IntentionNet

cfg = None
flags_obj = None

class MyModelCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, bestpath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, skip=1):
        super(MyModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.bestpath = bestpath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.skip = skip
        self.skip_count = 0
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        self.skip_count += 1
        if flags_obj.num_gpus > 1:
            # discard multi-gpu layer
            old_model = self.model.layers[-2]
        else:
            old_model = self.model

        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.bestpath
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            old_model.save_weights(filepath, overwrite=True)
                        else:
                            old_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))

        if self.skip_count >= self.skip:
            self.skip_count = 0
            filepath = self.filepath
            if self.verbose > 0:
                print('Epoch %05d: saving model to %s' % (epoch, filepath))
            if self.save_weights_only:
                old_model.save_weights(filepath, overwrite=True)
            else:
                old_model.save(filepath, overwrite=True)

def define_intention_net_flags():
    flags.DEFINE_enum(
            name='intention_mode', short_name='mode', default="DLM",
            enum_values=['DLM', 'LPE_SIAMESE', 'LPE_NO_SIAMESE'],
            help=help_wrap("Intention Net mode to run"))

    flags.DEFINE_enum(
            name='input_frame', short_name='input_frame', default="NORMAL",
            enum_values=['NORMAL', 'WIDE', 'MULTI'],
            help=help_wrap("Intention Net mode to run"))

    flags.DEFINE_enum(
            name='dataset', short_name='ds', default="PIONEER",
            enum_values=['CARLA_SIM', 'CARLA', 'HUAWEI', 'PIONEER', 'DUCKIETOWN'],
            help=help_wrap("dataset to load for training."))

    flags.DEFINE_boolean(
            name="segmented", short_name='seg', default=False,
            help=help_wrap("Specify if want to train with segmentation labels"))

    global cfg
    cfg = load_config(IntentionNetConfig)

def lr_schedule(epoch):
    """
    Learning rate schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.
    """
    lr = flags_obj.learning_rate
    if epoch > 90:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 5e-2
    elif epoch > 40:
        lr *= 1e-1
    elif epoch > 20:
        lr *= 5e-1
    print ('Learning rate: ', lr)
    return lr

def get_optimizer():
    if flags_obj.optim == 'rmsprop':
        optimizer = RMSprop(lr=flags_obj.learning_rate, decay=cfg.WEIGHT_DECAY, rho=0.9, epsilon=1e-08)
        print ('=> use rmsprop optimizer')
    elif flags_obj.optim == 'sgd':
        optimizer = SGD(lr=flags_obj.learning_rate, decay=cfg.WEIGHT_DECAY, momentum=cfg.MOMENTUM)
        print ('=> use sgd optimizer')
    elif flags_obj.optim == 'radam':
        optimizer = RAdam()
        print ('=> use RAdam optimizer')
    else:
        optimizer = Adam(lr=flags_obj.learning_rate, decay=cfg.WEIGHT_DECAY)
        print ('=> use adam optimizer')
    return optimizer

def weighted_mse(yTrue,yPred):
    # a = tf.placeholder(tf.float32)
    
    # sess = tf.Session()
    # with sess.as_default():
    #     # tensor = tf.range(10)
    #     print_op = tf.print(yPred)
    #     with tf.control_dependencies([print_op]):
    #         out = tf.add(yPred, yPred)
    #     sess.run(out)
    # assert False

    # with tf.Session() as sess:
    #     # init = tf.global_variables_initializer()
    #     # sess.run(init)
    #     print(f'yTrue: {yTrue.eval(feed_dict={a:1.0})} \n\n\n\n\n')
    # assert False
    factor = tf.constant([1, 3], dtype= tf.float32)
    sqe = tf.math.squared_difference(yTrue, yPred)
    wsqe = tf.math.multiply(sqe, factor)
    # theta = yPred[1]
    # factor = 5
    # weighted_angles = K.abs(theta) * factor * sqe

    # print(f'weighted_angles: {weighted_angles.eval()}')

    return K.mean(wsqe, axis=-1) # wtf axis=-1

def main(_):
    global flags_obj
    flags_obj = flags.FLAGS
    ###################################
    # TensorFlow wizardry
    #config = tf.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    #config.gpu_options.allow_growth = True
    # Only allow a total of half the GPU memory to be allocated
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # Create a session with the above options specified.
    #K.tensorflow_backend.set_session(tf.Session(config=config))
    ###################################

    if flags_obj.val_dir is None:
        flags_obj.val_dir = flags_obj.data_dir
    # get number of gpus, -1 means to use all gpus
    if flags_obj.num_gpus == -1:
        from tensorflow.python.client import device_lib  # pylint: disable=g-import-not-at-top
        local_device_protos = device_lib.list_local_devices()
        flags_obj.num_gpus = sum([1 for d in local_device_protos if d.device_type == "GPU"])
    print ("=> Using {} gpus".format(flags_obj.num_gpus))

    if flags_obj.dataset == 'CARLA':
        from dataset import CarlaImageDataset as Dataset
        print ('=> using CARLA published data')
    elif flags_obj.dataset == 'CARLA_SIM':
        from dataset import CarlaSimDataset as Dataset
        print ('=> using self-collected CARLA data')
    elif flags_obj.dataset == 'PIONEER':
        from dataset import PioneerDataset as Dataset
        print ('=> using pioneer data')
    elif flags_obj.dataset =="DUCKIETOWN":
        from dataset import DuckieTownDataset as Dataset
        print ('=> using duckietown data')
    else:
        from dataset import HuaWeiFinalDataset as Dataset
        print ('=> using HUAWEI data')

    print ('mode: ', flags_obj.mode, 'input frame: ', flags_obj.input_frame, 'batch_size', flags_obj.batch_size, cfg.NUM_INTENTIONS)
    model = IntentionNet(flags_obj.mode, flags_obj.input_frame, Dataset.NUM_CONTROL, cfg.NUM_INTENTIONS)

    if flags_obj.num_gpus > 1:
        # make the model parallel
        flags_obj.batch_size = flags_obj.batch_size * flags_obj.num_gpus
        model = multi_gpu_model(model, flags_obj.num_gpus)

    # print model summary
    model.summary()

    # optionally resume from a checkpoint
    if flags_obj.resume is not None:
        if os.path.isfile(flags_obj.resume):
            model.load_weights(flags_obj.resume)
            print ('=> loaded checkpoint {}'.format(flags_obj.resume))
        else:
            print ("=> no checkpoint found at {}".format(flags_obj.resume))

    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=1e-7)

    # save model
    best_model_fn = os.path.join(flags_obj.model_dir, flags_obj.input_frame + '_' + flags_obj.mode + '_best_model.h5')
    lastest_model_fn = os.path.join(flags_obj.model_dir, flags_obj.input_frame + '_' + flags_obj.mode + '_latest_model.h5')
    saveBestModel = MyModelCheckpoint(lastest_model_fn, best_model_fn, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', skip=10)
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()), write_graph=False, write_images=False, batch_size=flags_obj.batch_size)

    # callbacks
    callbacks = [saveBestModel, lr_reducer, lr_scheduler, tensorboard]

    # we choose max_samples to save time for training. For large dataset, we sample 200000 samples each epoch.
    train_generator = Dataset(flags_obj.data_dir, flags_obj.batch_size, cfg.NUM_INTENTIONS, mode=flags_obj.mode, shuffle=False, max_samples=200000, input_frame=flags_obj.input_frame, segmented=flags_obj.segmented)
    val_generator = Dataset(flags_obj.val_dir, flags_obj.batch_size, cfg.NUM_INTENTIONS, mode=flags_obj.mode, max_samples=1000, input_frame=flags_obj.input_frame, segmented=flags_obj.segmented)

    optimizer = get_optimizer()

    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy', 'mae'])

    model.fit_generator(
            generator=train_generator,
            validation_data=val_generator,
            use_multiprocessing=False,
            workers=flags_obj.num_workers,
            callbacks=callbacks,
            epochs=flags_obj.train_epochs)

if __name__ == '__main__':
    define_intention_net_flags()
    absl_app.run(main)
