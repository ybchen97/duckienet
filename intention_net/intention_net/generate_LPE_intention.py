"""This is a helper function to generate LPE intention from labeled dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app as absl_app
from absl import flags
import numpy as np
import matplotlib.image as mpimg
from matplotlib.patches import Circle
import cv2
import math
import os
from PIL import Image
from tqdm import tqdm

from main import define_intention_net_flags
from threadedgenerator import ThreadedGenerator
from toolz import partition_all
from config import *

cfg = None
flags_obj = None

def pos_to_pixel(carla_map, x, y):
    pixel = carla_map.convert_to_pixel([x, y, 0.22])
    return pixel

def generate_lpe_intention(intention_map, pixels, thetas, offset, files, max_plot_samples, line_thick=4, steps=120):
    lpes = []
    intention_map = cv2.copyMakeBorder(intention_map, offset, offset, offset, offset, cv2.BORDER_CONSTANT, value=[1, 1, 1, 1])
    it = ThreadedGenerator(range(len(pixels)), queue_maxsize=6400)
    for idx in partition_all(1, tqdm(it)):
        i = idx[0]
        pixel = pixels[i]
        theta = thetas[i]
        img = np.copy(intention_map[pixel[1]:pixel[1]+2*offset, pixel[0]:pixel[0]+2*offset])
        # add history
        for h in range(max(0, i-steps), i):
            h_pixel = (offset + pixels[h][0] - pixel[0], offset + pixels[h][1] - pixel[1])
            if h_pixel[0] > 0 and h_pixel[0] < offset*2 and h_pixel[1] > 0 and h_pixel[1] < offset*2:
                img[h_pixel[1]-line_thick:h_pixel[1]+line_thick, h_pixel[0]-line_thick:h_pixel[0]+line_thick] = [1.0, 0, 0, 1]
        # add ahead path
        for h in range(i, min(i+steps, len(pixels))):
            h_pixel = (offset + pixels[h][0] - pixel[0], offset + pixels[h][1] - pixel[1])
            if h_pixel[0] > 0 and h_pixel[0] < offset*2 and h_pixel[1] > 0 and h_pixel[1] < offset*2:
                img[h_pixel[1]-line_thick:h_pixel[1]+line_thick, h_pixel[0]-line_thick:h_pixel[0]+line_thick] = [0, 0, 1.0, 1]
        col, row, channel = img.shape
        M = cv2.getRotationMatrix2D((col/2, row/2), 90+theta, 1)
        img = cv2.warpAffine(img, M, (col, row), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 1)
        img = cv2.resize(img, (224, 224))

        mpimg.imsave(files[i], img)

        if i < max_plot_samples:
            lpes.append(img)
    return lpes

# radius in meters
def parse_carla_dataset(labels, carla_map, map_image, radius=20, max_plot_samples=1000):
    offset = pos_to_pixel(carla_map, radius, radius)[0] - pos_to_pixel(carla_map, 0, 0)[0]
    theta = np.arctan2(labels[:, -2], labels[:, -3]) * 180 / np.pi
    pixels = []
    files = []
    for i in range(labels.shape[0]):
        pixel = pos_to_pixel(carla_map, labels[i][1], labels[i][2])
        pixels.append(pixel)
        files.append(os.path.join(flags_obj.data_dir, 'lpe_'+str(int(labels[i,0]))+'.png'))

    lpes = generate_lpe_intention(map_image, pixels, theta, offset, files, max_plot_samples)

    plot_orientation(map_image, pixels, theta, lpes)

def plot_orientation(map_image, pixels, theta, lpes):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.imshow(map_image)
    ax2 = fig.add_subplot(222)
    ax2.plot(theta[:len(lpes)])
    ax3 = fig.add_subplot(223)
    for i in tqdm(range(len(pixels))):
        circle = Circle((pixels[i][0], pixels[i][1]), 1, color='r', label='A point')
        ax.add_patch(circle)
        if i % 50 == 0 and i < len(lpes):
            ax3.imshow(lpes[i])
            plt.draw()
            plt.pause(1e-5)
    plt.show()

def main(_):
    global flags_obj
    flags_obj = flags.FLAGS
    #assert (flags_obj.map is not None), "Must provide the map path for LPE"

    if flags_obj.dataset == 'CARLA':
        from dataset import CarlaImageDataset as Dataset
        labels = np.loadtxt(os.path.join(flags_obj.data_dir, 'label.txt'))
        img_ids = np.expand_dims(labels[:, 0], 1)
        labels = labels[:, 1:]
        # transfer from cm to meter.
        labels[:, [Dataset.POS_X, Dataset.POS_Y]] /= 100.0
        # parse useful properties
        labels = labels[:, [Dataset.POS_X, Dataset.POS_Y, Dataset.ORIENT_X, Dataset.ORIENT_Y, Dataset.ORIENT_Z]]
        labels = np.concatenate((img_ids, labels), axis=1)
        print ('=> using CARLA published data')
    elif flags_obj.dataset == 'CARLA_SIM':
        from dataset import CarlaSimDataset as Dataset
        print ('=> using self-collected CARLA data')
    else:
        print ('=> using HUAWEI data, the function is built-in the dataset')
        return

    print (labels.shape, "first 10 labels", labels[:10])

    # parse map for LPE
    if flags_obj.dataset.startswith('CARLA'):
        from carla.planner.map import CarlaMap
        PIXEL_DENSITY = 0.1653
        NODE_DENSITY = 50
        carla_map = CarlaMap('Town01', PIXEL_DENSITY, NODE_DENSITY)
        map_image = mpimg.imread(flags_obj.map)
        parse_carla_dataset(labels, carla_map, map_image)
    else:
        pass

if __name__ == '__main__':
    cfg = load_config(IntentionNetConfig)

    flags.DEFINE_enum(
            name='dataset', short_name='ds', default="CARLA",
            enum_values=['CARLA_SIM', 'CARLA', 'HUAWEI'],
            help=help_wrap("dataset to load for training."))

    # specified flags for global map to generate LPE intention
    flags.DEFINE_string(
        name='map', short_name='m', default='/home/gaowei/CARLA/PythonClient/carla/planner/Town01.png',
        help=help_wrap("Path to global map for LPE."))
    absl_app.run(main)
