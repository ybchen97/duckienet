"""config flags for intention net"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from easydict import EasyDict as edict

from absl import app as absl_app
from absl import flags
# misc
help_wrap = functools.partial(flags.text_wrap, length=80, indent="",
                              firstline_indent="\n")
absl_app.HelpshortFlag.SHORT_NAME = "h"
# base flags
flags.DEFINE_string(
    name="data_dir", short_name="dd", default="/path_to_repo/intention_net/data",
    help=help_wrap("The location of the input data."))

flags.DEFINE_string(
    name="val_dir", short_name="vd", default=None,
    help=help_wrap("The location of the validation data."))

flags.DEFINE_string(
    name="model_dir", short_name="md", default="/path_to_repo/intention_net/new_model",
    help=help_wrap("The location of the model checkpoint data."))

flags.DEFINE_string(
    name="export_dir", short_name="ed", default=None,
    help=help_wrap("If set, a SavedModel serialization of the model will "
		"be exported to this directory at the end of training. "
		"See the README for more details and relevant links."))

flags.DEFINE_boolean(
    name="evaluation", short_name='eval', default=False,
    help=help_wrap("Whether to directly evaluted the learned model"))

flags.DEFINE_string(
    name='resume', short_name='r', default='/home/duong/Downloads/train_data/NORMAL_DLM_latest_model.h5',
    help=help_wrap("Path to latest checkpoint for resume."))

flags.DEFINE_string(
    name="optim", short_name="otm", default="radam",
    help=help_wrap("Optimizer type for training"))

flags.DEFINE_integer(
    name="train_epochs", short_name="te", default=1,
    help=help_wrap("The number of epochs used to train"))

flags.DEFINE_integer(
    name="start_epoch", short_name="se", default=0,
    help=help_wrap("Manual epoch number (useful on restarts)"))

flags.DEFINE_integer(
    name="num_workers", short_name="nw", default=8,
    help=help_wrap("The number of data loading workers"))

flags.DEFINE_integer(
    name="epochs_between_evals", short_name="ebe", default=10,
    help=help_wrap("The number of training epochs to run between evaluations"))

flags.DEFINE_float('learning_rate', short_name="lr", default=1e-4,
    help=help_wrap('Initial learning rate.'))

flags.DEFINE_integer(
    name="batch_size", short_name="bs", default=16,
    help=help_wrap("Batch size for training and evaluation. When using "
		"multiple gpus, this is the global batch size for "
		"all devices. For example, if the batch size is 32 "
		"and there are 4 GPUs, each GPU will get 8 examples on "
		"each step."))

flags.DEFINE_integer(
    name="inter_op_parallelism_threads", short_name="inter", default=0,
    help=help_wrap("Number of inter_op_parallelism_threads to use for CPU. "
		"See TensorFlow config.proto for details."))

flags.DEFINE_integer(
    name="intra_op_parallelism_threads", short_name="intra", default=0,
    help=help_wrap("Number of intra_op_parallelism_threads to use for CPU. "
	    "See TensorFlow config.proto for details."))

flags.DEFINE_integer(
    name="num_gpus", short_name="ng",
    default=-1,
    help=help_wrap("How many GPUs to use with the DistributionStrategies API. The "
                    "default is 1 if TensorFlow can detect a GPU, and 0 otherwise."))

flags.DEFINE_integer(
    name="max_train_steps", short_name="mts", default=None, help=help_wrap(
	 "The model will stop training if the global_step reaches this "
	 "value. If not set, training will run until the specified number "
	 "of epochs have run as usual. It is generally recommended to set "
	 "--train_epochs=1 when using this flag."
	))

def get_conf_dict(cls):
    return {k: getattr(cls, k) for k in dir(cls) if not k.startswith('_')}

def load_config(cls):
    conf_dict = get_conf_dict(cls)
    for key, value in conf_dict.items():
        flags.FLAGS.set_default(name=key, value=value)
    return cls._C

class IntentionNetConfig(object):
    # default params
    train_epochs=15
    batch_size=16

    # Constants should start from _ in order not to get conflict with flags
    _C = edict()
    _C.NUM_INTENTIONS = 4
    # weight decay
    _C.WEIGHT_DECAY = 5e-5
    # momentum
    _C.MOMENTUM = 0.9
