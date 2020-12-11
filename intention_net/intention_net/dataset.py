""" dataset for intention net"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import os.path as osp
import csv
import math
import keras
import itertools
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.utils import to_categorical
from glob import glob
from tqdm import tqdm

class BaseDataset(keras.utils.Sequence):
    NUM_CONTROL = 2
    def __init__(self, data_dir, batch_size, num_intentions, mode, target_size=(224, 224), shuffle=False, max_samples=None, preprocess=True, input_frame='NORMAL'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_intentions = num_intentions
        self.mode = mode
        self.target_size = target_size
        self.shuffle = shuffle
        self.max_samples = max_samples
        # whether to preprocess image
        self.preprocess = preprocess
        self.input_frame = input_frame
        self.num_samples = None

        self.init()

        assert (self.num_samples is not None), "You must assign self.num_samples in init() function."

        if self.max_samples is not None:
            self.num_samples = min(self.max_samples, self.num_samples)

        self.on_epoch_end()

    def init(self):
        # you must assigh the num_samples here.
        raise NotImplementedError

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(self.num_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Denote number of batches per epoch"""
        return self.num_samples // self.batch_size

class PioneerDataset(BaseDataset):
    INTENTION_MAPPING = {
        'left' : 1,
        'right': 2,
        'forward':0,
        'stop':0,
    }
    INTENTION_MAPPING_NAME = {
        1: 'left',
        2: 'right',
        0: 'forward',
        3: 'stop'
    }
    # only add dlm support here, you can extend to lpe easily refering to the HuaweiFinalDataset example
    NUM_CONTROL = 2

    # use to normalize regression data
    SCALE_VEL = 0.5
    SCALE_STEER = 0.5

    def __init__(self, data_dir, batch_size, num_intentions, mode, target_size=(224, 224), shuffle=False, max_samples=None, preprocess=True, input_frame='NORMAL'):
        super().__init__(data_dir, batch_size, num_intentions, mode, target_size, shuffle, max_samples, preprocess, input_frame)

    def init(self):
        base_dir = osp.join(self.data_dir, 'data')
        self.data_header, self.data = self.read_csv(os.path.join(base_dir, 'label.txt'))
        self.num_samples = len(self.data)

        # reverse header index
        self.data_idx = {}
        for k, v in enumerate(self.data_header):
            self.data_idx[v] = k

        self.images = []
        for datum in self.data:
            if self.input_frame == 'MULTI': # front + left + right cam
                fn_left = osp.join(base_dir, 'rgb_6', f"{int(datum[self.data_idx['frame']])}.jpg")
                fn_right = osp.join(base_dir, 'rgb_4', f"{int(datum[self.data_idx['frame']])}.jpg")
                fn_front = osp.join(base_dir, 'rgb_0', f"{int(datum[self.data_idx['frame']])}.jpg")
                self.images.append([fn_left,fn_front,fn_right])
            else: # front cam 
                fn = osp.join(base_dir, 'rgb_0', f"{int(datum[self.data_idx['frame']])}.jpg")
                self.images.append(fn)

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = []
        I = []
        Y = []
        XL = []
        XM = []
        XR = []
        for idx in indexes:
            lbl = self.data[idx]
            
            if self.input_frame == 'MULTI': # front + left + right cam
                left_img = img_to_array(load_img(self.images[idx][0], target_size=self.target_size))
                front_img = img_to_array(load_img(self.images[idx][1], target_size=self.target_size))
                right_img = img_to_array(load_img(self.images[idx][2], target_size=self.target_size))
                if self.preprocess:
                    front_img = preprocess_input(front_img)
                    left_img = preprocess_input(left_img)
                    right_img = preprocess_input(right_img)
                XL.append(left_img)
                XM.append(front_img)
                XR.append(right_img)
            else: # only use front camera
                img = img_to_array(load_img(self.images[idx], target_size=self.target_size))
                if self.preprocess:
                    img = preprocess_input(img)
                X.append(img)

            if self.mode == 'DLM':
                #lbl_intention = lbl[self.data_idx['dlm']]
                lbl_intention = self.INTENTION_MAPPING[lbl[self.data_idx['dlm']]]
                intention = to_categorical(lbl_intention, num_classes=self.num_intentions)

            control = [float(lbl[self.data_idx['current_velocity']])/self.SCALE_VEL, (float(lbl[self.data_idx['steering_wheel_angle']]))/self.SCALE_STEER]
            I.append(intention)
            Y.append(control)
        if self.input_frame != 'MULTI':
            X = np.array(X)
            I = np.array(I)
            Y = np.array(Y)
            return [X, I], Y
        else:
            XL = np.array(XL)
            XM = np.array(XM)
            XR = np.array(XR)
            I = np.array(I)
            Y = np.array(Y)
            return [XL,XM,XR,I],Y

    def read_csv(self, fn, has_header=True):
        f = open(fn)
        reader = csv.reader(f, delimiter=' ')
        header = None
        data = []
        if has_header:
            row_num = 0
            for row in reader:
                if row_num == 0:
                    header = row
                    row_num += 1
                else:
                    data.append(row)
                    row_num += 1
        else:
            for row in reader:
                data.append(row)

        # drop the last row because sometimes the last row is not complete
        return header, data[:-1]

class CarlaSimDataset(BaseDataset):
    # intention mapping
    INTENTION_MAPPING = {}
    INTENTION_MAPPING[0] = 0
    INTENTION_MAPPING[2] = 1
    INTENTION_MAPPING[3] = 2
    INTENTION_MAPPING[4] = 3
    INTENTION_MAPPING[5] = 4

    NUM_CONTROL = 2
    def __init__(self, data_dir, batch_size, num_intentions, mode, target_size=(224, 224), shuffle=False, max_samples=None, preprocess=True, input_frame='NORMAL'):
        super().__init__(data_dir, batch_size, num_intentions, mode, target_size, shuffle, max_samples, preprocess, input_frame)

    def init(self):
        self.labels = []
        self.files = []
        image_path_pattern = '_images/episode_{:s}/{:s}/image_{:0>5d}.jpg.png'
        frames = {}
        with open(osp.join(self.data_dir, 'measurements.csv'), 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.labels.append(row)
                episode_name = row['weather'] + '_' + row['exp_id'] + '_' + row['start_point'] + '.' + row['end_point']
                if episode_name in frames:
                    frames[episode_name] += 1
                else:
                    frames[episode_name] = 0
                fn = image_path_pattern.format(episode_name, 'CameraRGB', frames[episode_name])
                self.files.append(osp.join(self.data_dir, fn))
        self.num_samples = len(self.labels)

    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = []
        I = []
        S = []
        Y = []
        for idx in indexes:
            lbl = self.labels[idx]
            img = load_img(self.files[idx], target_size=self.target_size)
            img = preprocess_input(img_to_array(img))
            intention = to_categorical(self.INTENTION_MAPPING[int(float(lbl['intention']))], num_classes=self.num_intentions)
            speed = [float(lbl['speed'])]
            control = [float(lbl['steer']), float(lbl['throttle'])-float(lbl['brake'])]
            X.append(img)
            I.append(intention)
            S.append(speed)
            Y.append(control)
        X = np.array(X)
        I = np.array(I)
        S = np.array(S)
        Y = np.array(Y)
        return [X, I, S], Y

class CarlaImageDataset(CarlaSimDataset):
    STEER = 0
    GAS = 1
    BRAKE = 2
    HAND_BRAKE = 3
    REVERSE_GEAR = 4
    STEER_NOISE = 5
    GAS_NOISE = 6
    BRAKE_NOISE = 7
    POS_X = 8
    POS_Y = 9
    SPEED = 10
    COLLISION_OTHER = 11
    COLLISION_PEDESTRIAN = 12
    COLLISION_CAR = 13
    OPPOSITE_LANE_INTER = 14
    SIDEWALK_INTER = 15
    ACC_X = 16
    ACC_Y = 17
    ACC_Z = 18
    PLATFORM_TIME = 19
    GAME_TIME = 20
    ORIENT_X = 21
    ORIENT_Y = 22
    ORIENT_Z = 23
    INTENTION = 24
    NOISE = 25
    CAMERA = 26
    CAMERA_YAW = 27

    def __init__(self, data_dir, batch_size, num_intentions, mode, target_size=(224, 224), shuffle=False, max_samples=None, preprocess=True, input_frame='NORMAL'):
        super().__init__(data_dir, batch_size, num_intentions, mode, target_size, shuffle, max_samples, preprocess, input_frame)

    def init(self):
        self.labels = np.loadtxt(osp.join(self.data_dir, 'label.txt'))
        self.num_samples = self.labels.shape[0]

        self.files = [self.data_dir + '/' + str(fn)+'.png' for fn in self.labels[:,0].astype(np.int32)][:self.num_samples]
        if self.mode.startswith('LPE'):
            self.lpe_files = [self.data_dir + '/lpe_' + str(fn)+'.png' for fn in self.labels[:,0].astype(np.int32)][:self.num_samples]

        self.labels = self.labels[:,1:]

    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = []
        I = []
        S = []
        Y = []
        for idx in indexes:
            lbl = self.labels[idx]
            img = load_img(self.files[idx], target_size=self.target_size)
            img = preprocess_input(img_to_array(img))
            if self.mode == 'DLM':
                intention = to_categorical(self.INTENTION_MAPPING[lbl[self.INTENTION]], num_classes=self.num_intentions)
            else:
                intention = load_img(self.lpe_files[idx], target_size=self.target_size)
                intention = preprocess_input(img_to_array(intention))
            # transfer from km/h to m/s
            speed = [lbl[self.SPEED]/3.6]
            control = [lbl[self.STEER], lbl[self.GAS]-lbl[self.BRAKE]]
            X.append(img)
            I.append(intention)
            S.append(speed)
            Y.append(control)
        X = np.array(X)
        I = np.array(I)
        S = np.array(S)
        Y = np.array(Y)
        return [X, I, S], Y

class HuaWeiFinalDataset(BaseDataset):
    STRAIGHT_FORWARD = 0
    STRAIGHT_BACK = 1
    LEFT_TURN = 2
    RIGHT_TURN = 3
    LANE_FOLLOW = 4
    # intention wrapper
    INTENTION = {
            STRAIGHT_FORWARD: 0,
            LEFT_TURN: 1,
            RIGHT_TURN: 2
            }
    # use to normalize regression data
    SCALE_ACC = 0.8
    SCALE_STEER = 2*np.pi

    def __init__(self, data_dir, batch_size, num_intentions, mode, target_size=(224, 224), shuffle=False, max_samples=None, preprocess=True, input_frame='NORMAL'):
        super().__init__(data_dir, batch_size, num_intentions, mode, target_size, shuffle, max_samples, preprocess, input_frame)

    def init(self):
        routes = glob(os.path.join(self.data_dir, 'route*'))
        self.raw_labels = []
        assert (len(routes) > 0), "no routes data founded in folder {}".format(self.data_dir)
        for route in routes:
            self.car_data_header, self.car_data = self.read_csv(os.path.join(route, 'LabelData_VehicleData_PRT.txt'))
            self.raw_labels.append(self.car_data)
        # reverse header index
        self.car_data_idx = {}
        for k, v in enumerate(self.car_data_header):
            self.car_data_idx[v] = k

        # sync data
        self.labels = []
        self.images = []
        self.lpes = []
        self.num_samples = 0
        for i, label in enumerate(self.raw_labels):
            # we drop the first few stop images when starting, because it is not a valid label
            valid_start = False
            labeled_data = []
            labeled_images = []
            labeled_lpes = []
            print (routes[i])
            for data in label:
                if float(data[self.car_data_idx['current_velocity']]) > 0 and valid_start == False:
                    valid_start = True
                if valid_start:
                    if self.input_frame == 'NORMAL':
                        fn = os.path.join(routes[i], 'camera_img/front_60/{}.jpg'.format(int(data[self.car_data_idx['img_front_60_frame']])))
                    elif self.input_frame == 'WIDE':
                        fn = os.path.join(routes[i], 'camera_img/front_96_left/{}.jpg'.format(int(data[self.car_data_idx['img_front_60_frame']])))
                    else:
                        fn_l = os.path.join(routes[i], 'camera_img/side_96_left/{}.jpg'.format(int(data[self.car_data_idx['img_front_60_frame']])))
                        fn_m = os.path.join(routes[i], 'camera_img/front_60/{}.jpg'.format(int(data[self.car_data_idx['img_front_60_frame']])))
                        fn_r = os.path.join(routes[i], 'camera_img/side_96_right/{}.jpg'.format(int(data[self.car_data_idx['img_front_60_frame']])))
                        fn = [fn_l, fn_m, fn_r]
                    labeled_images.append(fn)
                    lpe_fn = os.path.join(routes[i], 'intention_img/{}.jpg'.format(int(data[self.car_data_idx['intention_img']])))
                    labeled_lpes.append(lpe_fn)
                    labeled_data.append(data)
            self.num_samples += len(labeled_images)
            self.labels.append(labeled_data)
            self.images.append(labeled_images)
            self.lpes.append(labeled_lpes)

        self.list_labels = list(itertools.chain.from_iterable(self.labels))
        self.list_images = list(itertools.chain.from_iterable(self.images))
        self.list_lpes = list(itertools.chain.from_iterable(self.lpes))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = []
        XL = []
        XM = []
        XR = []
        I = []
        S = []
        Y = []
        for idx in indexes:
            lbl = self.list_labels[idx]
            if self.input_frame == 'MULTI':
                img_l = img_to_array(load_img(self.list_images[idx][0], target_size=self.target_size))
                img_m = img_to_array(load_img(self.list_images[idx][1], target_size=self.target_size))
                img_r = img_to_array(load_img(self.list_images[idx][2], target_size=self.target_size))
                img = [img_l, img_m, img_r]
                if self.preprocess:
                    img = [preprocess_input(im) for im in img]
                XL.append(img[0])
                XM.append(img[1])
                XR.append(img[2])
            else:
                img = img_to_array(load_img(self.list_images[idx], target_size=self.target_size))
                if self.preprocess:
                    img = preprocess_input(img)
                X.append(img)

            if self.mode == 'DLM':
                # add data augmentation
                lbl_intention = self.INTENTION[int(lbl[self.car_data_idx['intention_type']])]
                if self.preprocess:
                    if float(lbl[self.car_data_idx['steering_wheel_angle']]) < 0.05:
                        lbl_intention = np.random.randint(self.num_intentions)
                intention = to_categorical(lbl_intention, num_classes=self.num_intentions)
            else:
                intention = img_to_array(load_img(self.list_lpes[idx], target_size=self.target_size))
                if self.preprocess:
                    intention = preprocess_input(intention)

            extra = [float(lbl[self.car_data_idx['current_velocity']])]
            control = [float(lbl[self.car_data_idx['steering_wheel_angle']])/self.SCALE_STEER, (float(lbl[self.car_data_idx['ax']])+0.4)/self.SCALE_ACC]

            I.append(intention)
            S.append(extra)
            Y.append(control)

        if self.input_frame == 'MULTI':
            XL = np.array(XL)
            XM = np.array(XM)
            XR = np.array(XR)
            I = np.array(I)
            S = np.array(S)
            Y = np.array(Y)
            return [XL, XM, XR, I, S], Y
        else:
            X = np.array(X)
            I = np.array(I)
            S = np.array(S)
            Y = np.array(Y)
            return [X, I, S], Y

    def read_csv(self, fn, has_header=True):
        f = open(fn)
        reader = csv.reader(f, delimiter=' ')
        header = None
        data = []
        if has_header:
            row_num = 0
            for row in reader:
                if row_num == 0:
                    header = row
                    row_num += 1
                else:
                    data.append(row)
                    row_num += 1
        else:
            for row in reader:
                data.append(row)

        # drop the last row because sometimes the last row is not complete
        return header, data[:-1]

class HuaWeiDataset(BaseDataset):
    TURN_RIGHT = 0
    GO_STRAIGHT = 1
    TURN_LEFT = 2
    REACH_GOAL = 3
    DEFAULT_INTENTION = GO_STRAIGHT

    def __init__(self, data_dir, batch_size, num_intentions, mode, target_size=(224, 224), shuffle=False, max_samples=None, preprocess=True):
        super().__init__(data_dir, batch_size, num_intentions, mode, target_size, shuffle, max_samples, preprocess)

    def init(self):
        routes = glob(os.path.join(self.data_dir, 'Log*'))
        self.org_labels = []
        for route in routes:
            self.car_data_header, self.car_data = self.read_csv(os.path.join(route, 'LabelData_VehicleData_PRT.txt'))
            self.org_labels.append(self.car_data)
        # reverse header index
        self.car_data_idx = {}
        for k, v in enumerate(self.car_data_header):
            self.car_data_idx[v] = k
        self.global_map = None

        # sync data
        self.labels = []
        self.images = []
        self.intentions = []
        self.num_samples = 0
        for i, label in enumerate(self.org_labels):
            # we drop the first few stop images when starting, because it is not a valid label
            valid_start = False
            labeled_images = []
            labeled_data = []
            labeled_lpes = []
            for data in label:
                if float(data[self.car_data_idx['current_velocity']]) > 0 and valid_start == False:
                    valid_start = True
                if valid_start:
                    fn = os.path.join(routes[i], 'LabelImages/{}.jpg'.format(
                        int(data[self.car_data_idx['img_frame']])))
                    labeled_images.append(fn)
                    lpe_fn = os.path.join(routes[i], 'LabelImages/lpe_{}.png'.format(
                        int(data[self.car_data_idx['img_frame']])))
                    labeled_lpes.append(lpe_fn)
                    labeled_data.append(data)
            self.num_samples += len(labeled_images)
            self.labels.append(labeled_data)
            self.images.append(labeled_images)
            self.intentions.append(labeled_lpes)

        self.files = list(itertools.chain.from_iterable(self.images))
        self.lpe_files = list(itertools.chain.from_iterable(self.intentions))
        self.car_labels = list(itertools.chain.from_iterable(self.labels))
        if self.mode == 'DLM':
            self.generate_dlm()

    def get_global_map(self, lat_min, lon_min, lat_max, lon_max, z=18):
        import pyMap
        import matplotlib.image as mpimg
        self.left, self.top = pyMap.latlng2tilenum(lat_max, lon_min, z)
        fn = os.path.join(os.path.dirname(__file__), 'output', 'huawei.png')
        if not os.path.isfile(fn):
            pyMap.process_latlng(lat_max, lon_min, lat_min, lon_max, z, output='huawei', maptype='gaode')
        self.global_map = mpimg.imread(fn)

    def latlng2pixel(self, lat_deg, lng_deg, zoom=18):
        n = math.pow(2, int(zoom))
        xtile = ((lng_deg + 180) / 360) * n
        lat_rad = lat_deg / 180 * np.pi
        ytile = (1 - (np.log(np.tan(lat_rad) + 1 / np.cos(lat_rad)) / np.pi)) / 2 * n
        x = (xtile - self.left)*256
        y = (ytile - self.top)*256
        return (np.floor(x).astype(np.int32), np.floor(y).astype(np.int32))

    def get_pixels(self):
        #convert from gcj2 to wgs-84
        from coord_convert.utils import Transform
        transform = Transform()
        longitudes = []
        latitudes = []
        thetas = []
        for label in self.labels:
            lats = []
            lons = []
            ths = []
            for data in label:
                lon = float(data[self.car_data_idx['longitude']])
                lat = float(data[self.car_data_idx['latitude']])
                wgs_lon, wgs_lat = transform.wgs2gcj(lon, lat)
                lons.append(wgs_lon)
                lats.append(wgs_lat)
                ths.append(-float(data[self.car_data_idx['absolute_heading']])*180/np.pi)
            latitudes.append(np.array(lats))
            longitudes.append(np.array(lons))
            thetas.append(np.array(ths))
        lat_min = min([a.min() for a in latitudes])
        lat_max = max([a.max() for a in latitudes])
        lon_min = min([a.min() for a in longitudes])
        lon_max = max([a.max() for a in longitudes])
        self.get_global_map(lat_min, lon_min, lat_max, lon_max)
        pixels = [self.latlng2pixel(lats, lons) for lats, lons in zip(latitudes, longitudes)]
        return pixels, thetas

    def generate_dlm(self, lookahead_steps=300, turning_threshold=20):
        pixels, thetas = self.get_pixels()
        self.dlms = []
        for r, pixs in enumerate(tqdm(pixels)):
            # look ahead orientation
            for idx in tqdm(range(len(thetas[r]))):
                current_theta = thetas[r][idx]
                lookahead_theta = []
                for l in range(idx, min(idx+lookahead_steps, len(thetas[r]))):
                    lookahead_theta.append(thetas[r][l]-current_theta)
                turning_angle = np.mean(lookahead_theta)
                if len(lookahead_theta) < lookahead_steps:
                    self.dlms.append(self.REACH_GOAL)
                elif turning_angle < -turning_threshold:
                    self.dlms.append(self.TURN_LEFT)
                elif turning_angle > turning_threshold:
                    self.dlms.append(self.TURN_RIGHT)
                else:
                    self.dlms.append(self.GO_STRAIGHT)

    def generate_lpe(self):
        """
        Use to generate lpe intention for huawei data.
        We move the function from generate_LPE_intention.py to here for simplicity
        """
        from generate_LPE_intention import generate_lpe_intention, plot_orientation
        pixels, thetas = self.get_pixels()
        for r, pixs in enumerate(pixels):
            pixs = list(zip(pixs[0], pixs[1]))
            lpes = generate_lpe_intention(self.global_map, pixs, thetas[r], 30, self.intentions[r], 3000, line_thick=2, steps=300)
        plot_orientation(self.global_map, pixs, thetas[-1], lpes)

    # for debug use
    def plot_trajectory(self):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Arc
        pixels, thetas = self.get_pixels()
        offset = 0
        for r, pixs in enumerate(pixels):
            x, y = pixs
            ax = plt.subplot(111)
            ax.imshow(self.global_map)
            ax.plot(x, y, 'r-')
            skip = 100
            arc_radius = 8
            for i, theta in enumerate(thetas[r]):
                if i % skip == 0:
                    ax.arrow(x[i], y[i],
                            0.05 * np.cos(theta/180*np.pi),
                            0.05 * np.sin(theta/180*np.pi),
                            head_width=1, head_length=20, fc='k', ec='k')
                    """
                    arc = Arc((x[i], y[i]),
                      arc_radius*2, arc_radius*2,  # ellipse width and height
                      theta1=0, theta2=theta*180/np.pi, linestyle='dashed')
                    ax.add_patch(arc)
                    """
                    if self.mode == 'DLM':
                        intention = self.dlms[i + offset]
                        ax.arrow(x[i], y[i],
                                0.5 * np.cos(-intention*np.pi/2),
                                0.5 * np.sin(-intention*np.pi/2),
                                head_width=3, head_length=20, fc='b', ec='b')
            offset += len(thetas[r])
        plt.show()

    def read_csv(self, fn, has_header=True):
        f = open(fn)
        reader = csv.reader(f, delimiter=' ')
        header = None
        data = []
        if has_header:
            row_num = 0
            for row in reader:
                if row_num == 0:
                    header = row
                    row_num += 1
                else:
                    data.append(row)
                    row_num += 1
        else:
            for row in reader:
                data.append(row)

        return header, data

    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = []
        I = []
        S = []
        Y = []
        for idx in indexes:
            lbl = self.car_labels[idx]
            img = load_img(self.files[idx], target_size=self.target_size)
            img = preprocess_input(img_to_array(img))
            if self.mode == 'DLM':
                intention = to_categorical(self.dlms[idx], num_classes=self.num_intentions)
            else:
                intention = load_img(self.lpe_files[idx], target_size=self.target_size)
                intention = preprocess_input(img_to_array(intention))
            # transfer from km/h to m/s
            speed = [float(lbl[self.car_data_idx['current_velocity']])]
            control = [np.pi/180.0*float(lbl[self.car_data_idx['steering_wheel_angle']]), float(lbl[self.car_data_idx['ax']])]
            X.append(img)
            I.append(intention)
            S.append(speed)
            Y.append(control)
        X = np.array(X)
        I = np.array(I)
        S = np.array(S)
        Y = np.array(Y)
        return [X, I, S], Y

intention_mapping = CarlaSimDataset.INTENTION_MAPPING

class DuckieTownDataset(BaseDataset):
    STRAIGHT_FORWARD = 0
    STRAIGHT_BACK = 1
    LEFT_TURN = 2
    RIGHT_TURN = 3
    LANE_FOLLOW = 4
    # intention wrapper
    INTENTION = {
            STRAIGHT_FORWARD: 0,
            LEFT_TURN: 1,
            RIGHT_TURN: 2
            }
    # use to normalize regression data
    SCALE_ACC = 0.8
    SCALE_STEER = 2*np.pi

    def __init__(self, data_dir, batch_size, num_intentions, mode, target_size=(224, 224), shuffle=False, max_samples=None, preprocess=True, input_frame='NORMAL', segmented=False):
        self.segmented = segmented
        super().__init__(data_dir, batch_size, num_intentions, mode, target_size, shuffle, max_samples, preprocess, input_frame)

    def init(self):
        if self.segmented:
            self.list_images = glob(os.path.join(self.data_dir, 'labels/L_*'))
            print(self.list_images[0])
        else:
            self.list_images = glob(os.path.join(self.data_dir, 'images/X_*'))
        self.list_lpes = glob(os.path.join(self.data_dir, 'intentions/I_*'))
        self.list_labels = glob(os.path.join(self.data_dir, 'actions/Y_*'))
        self.num_samples = len(self.list_images)

        # print(f"num_samples: {self.num_samples}") # TODO: delete
        assert len(self.list_images) == len(self.list_labels), "Number of image samples must tally with number of action samples; i "

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = []
        XL = []
        XM = []
        XR = []
        I = []
        S = []
        Y = []
        for idx in indexes:
            # print("\nSelected path: ", self.list_images[idx])

            # Retrieve image number
            # Use this image number to retrieve corresponding action and intention
            file_name = self.list_images[idx]
            if self.segmented:
                img_num = file_name[file_name.index("L") + 2: file_name.index(".")]
            else:
                img_num = file_name[file_name.index("X") + 2: file_name.index(".")]
            
            # print(f"Image num: {img_num}")

            # Load action
            lbl = os.path.join(self.data_dir, f"actions/Y_{img_num}.npy")
            # print(f"Action path: {lbl}")

            # Load image
            if self.input_frame == 'MULTI':
                assert False, "MULTI not supported for DuckyTown"
                img_l = img_to_array(load_img(self.list_images[idx][0], target_size=self.target_size))
                img_m = img_to_array(load_img(self.list_images[idx][1], target_size=self.target_size))
                img_r = img_to_array(load_img(self.list_images[idx][2], target_size=self.target_size))
                img = [img_l, img_m, img_r]
                if self.preprocess:
                    img = [preprocess_input(im) for im in img]
                XL.append(img[0])
                XM.append(img[1])
                XR.append(img[2])
            else:
                if self.segmented:
                    image_path = os.path.join(self.data_dir, f"labels/L_{img_num}.png")

                    # load_img automatically duplicates the label values along the last axis to give (224,224,3)
                    self.preprocess = False
                else:
                    image_path = os.path.join(self.data_dir, f"images/X_{img_num}.png")

                img = img_to_array(load_img(image_path, target_size=self.target_size))
                # print(f"Image path: {image_path}")

                if self.preprocess:
                    img = preprocess_input(img)
                X.append(img)

            # Load intention
            if self.mode == 'DLM':
                assert False, "DLM not supported for Duckytown"
                # add data augmentation
                lbl_intention = self.INTENTION[int(lbl[self.car_data_idx['intention_type']])]
                if self.preprocess:
                    if float(lbl[self.car_data_idx['steering_wheel_angle']]) < 0.05:
                        lbl_intention = np.random.randint(self.num_intentions)
                intention = to_categorical(lbl_intention, num_classes=self.num_intentions)
            else:
                intention_path = os.path.join(self.data_dir, f"intentions/I_{img_num}.png")
                intention = img_to_array(load_img(intention_path, target_size=self.target_size))
                # print(f"Intention path: {intention_path}")
                if self.preprocess:
                    intention = preprocess_input(intention)

            I.append(intention)
            S.append([0]) # dummy input
            # print(f'lbl: {lbl}')
            Y.append(np.load(lbl)) # (velocity, steering angle)

        if self.input_frame == 'MULTI':
            XL = np.array(XL)
            XM = np.array(XM)
            XR = np.array(XR)
            I = np.array(I)
            S = np.array(S)
            Y = np.array(Y)
            return [XL, XM, XR, I, S], Y
        else:
            X = np.array(X)
            I = np.array(I)
            S = np.array(S)
            Y = np.array(Y)
            return [X, I, S], Y

    def read_csv(self, fn, has_header=True):
        f = open(fn)
        reader = csv.reader(f, delimiter=' ')
        header = None
        data = []
        if has_header:
            row_num = 0
            for row in reader:
                if row_num == 0:
                    header = row
                    row_num += 1
                else:
                    data.append(row)
                    row_num += 1
        else:
            for row in reader:
                data.append(row)

        # drop the last row because sometimes the last row is not complete
        return header, data[:-1]

def test():
    #d = CarlaSimDataset('/home/gaowei/SegIRLNavNet/_benchmarks_results/Debug', 2, 5, max_samples=10)
    #d = CarlaImageDataset('/media/gaowei/Blade/linux_data/carla_data/AgentHuman/ImageData', 2, 5, mode='LPE_SIAMESE', max_samples=10)
    #d = HuaWeiDataset('/media/gaowei/Blade/linux_data/HuaWeiData', 2, 5, 'DLM', max_samples=10)
    d = HuaWeiFinalDataset('/home/gaowei/Data/Huawei/data/train/data', 2, 5, 'LPE_SIAMESE', max_samples=10, preprocess=False, input_frame='MULTI')
    #d.generate_lpe()
    #d.plot_trajectory()
    import matplotlib.pyplot as plt
    for step, (x,y) in enumerate(d):
        print (x[0].shape, x[1].shape, x[2].shape, y.shape)
        print (x[2], y)
        plt.imshow(x[0][0])
        plt.show()
        if step == len(d)-1:
            break

def check_valid_data():
    d = HuaWeiFinalDataset('/data/gaowei/huawei/Data', 2, 5, 'LPE_SIAMESE', preprocess=False, input_frame='MULTI')
    for step, (x,y) in enumerate(tqdm(d)):
        if step == len(d)-1:
            break

#test()
#check_valid_data()
