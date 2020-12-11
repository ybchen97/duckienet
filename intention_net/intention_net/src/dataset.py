import torch
from torch.utils.data import Dataset
import os.path as osp
import os 
from PIL import Image
from skimage import io, transform
from skimage.color import rgb2gray
import numpy as np 

def read_csv(path):
    data = dict()
    label = dict()
    with open(path,'r') as file:
        lines = file.readlines()[1:]
        for line in lines:
            line = line[:-1]
            tmp = line.split(" ")
            data[tmp[0]] = tmp[4]#remove \n
            label[tmp[0]] = [float(tmp[2]),float(tmp[3])]
    return data,label

class MultiCamPioneerDataset(Dataset):
    SCALE_VEL = 0.5
    SCALE_STEER = 0.5
    INTENTION_MAPPING = {
        'left' : 1,
        'right': 2,
        'forward':0,
        'stop':3,
    }

    def __init__(self,data_dir,target_size=(224,224),transform=None,BASE_DIR='',LEFT_GRAYSCALE_DIR='rgb_2',RIGHT_GRAYSCALE_DIR='rgb_1',FRONT_GRAYSCALE_DIR='rgb_0',LEFT_DEPTH_DIR='depth_2',RIGHT_DEPTH_DIR='depth_1',FRONT_DEPTH_DIR='depth_0'):
        super(MultiCamPioneerDataset,self).__init__()
        self.data_dir = data_dir
        self.target_size = target_size
        self.transform = transform
        self.BASE_DIR = BASE_DIR
        self.LEFT_GRAYSCALE_DIR = LEFT_GRAYSCALE_DIR
        self.FRONT_GRAYSCALE_DIR = FRONT_GRAYSCALE_DIR
        self.RIGHT_GRAYSCALE_DIR = RIGHT_GRAYSCALE_DIR
        self.LEFT_DEPTH_DIR = LEFT_DEPTH_DIR
        self.FRONT_DEPTH_DIR = FRONT_DEPTH_DIR
        self.RIGHT_DEPTH_DIR = RIGHT_DEPTH_DIR

        base_dir = osp.join(self.data_dir, self.BASE_DIR)
        _data,_label = read_csv(os.path.join(base_dir, 'label.txt')) # data: map img_fn -> intention, label: map img_fn->(vel,turning_angle)
        all_fn = list(_data.keys())
        valid_fn = list(filter(lambda x: self._filter_dataset(x),all_fn)) # only keep fn that has 6 images
        
        self.data = {k:_data[k] for k in valid_fn}
        self.label = {k:_label[k] for k in valid_fn}
        self.index = list(self.data.keys()) # list of img
        self.num_samples = len(self.data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self,idx):
        # get file name 
        fn = self.index[idx]
        intention = self.INTENTION_MAPPING[self.data[fn]]

        lbnw,mbnw,rbnw,dl,dm,dr = self._get_img(fn)
        lbnw,mbnw,rbnw,dl,dm,dr = self._preprocess(lbnw,mbnw,rbnw,dl,dm,dr)
        intention = torch.tensor(intention).long()
        
        target = torch.tensor(self.label[fn]).float()
        
        return [intention,dl,dm,dr,lbnw,mbnw,rbnw],target
    
    def _get_fn(self,fn):
        lbnw_im_path = osp.join(self.data_dir,self.BASE_DIR,self.LEFT_GRAYSCALE_DIR,fn+".jpg")
        mbnw_im_path = osp.join(self.data_dir,self.BASE_DIR,self.FRONT_GRAYSCALE_DIR,fn+".jpg")
        rbnw_im_path = osp.join(self.data_dir,self.BASE_DIR,self.RIGHT_GRAYSCALE_DIR,fn+".jpg")
        dl_im_path = osp.join(self.data_dir,self.BASE_DIR,self.LEFT_DEPTH_DIR,fn+".jpg")
        dm_im_path = osp.join(self.data_dir,self.BASE_DIR,self.FRONT_DEPTH_DIR,fn+".jpg")
        dr_im_path = osp.join(self.data_dir,self.BASE_DIR,self.RIGHT_DEPTH_DIR,fn+".jpg")

        return lbnw_im_path,mbnw_im_path,rbnw_im_path,dl_im_path,dm_im_path,dr_im_path

    def _get_img(self,fn):
        lbnw_im_path,mbnw_im_path,rbnw_im_path,dl_im_path,dm_im_path,dr_im_path = self._get_fn(fn)
        
        lbnw = np.array(Image.open(lbnw_im_path))/255.0
        mrgb = np.array(Image.open(mbnw_im_path))
        mbnw = rgb2gray(mrgb)
        rbnw = np.array(Image.open(rbnw_im_path))/255.0

        dl = np.array(Image.open((dl_im_path)))/255.0
        dm = np.array(Image.open((dm_im_path)))/255.0
        dr = np.array(Image.open((dr_im_path)))/255.0

        return lbnw,mbnw,rbnw,dl,dm,dr
    
    def _preprocess(self,lbnw,mbnw,rbnw,dl,dm,dr):
        if self.transform:
            lbnw = self.transform(lbnw)
            mbnw = self.transform(mbnw)
            rbnw = self.transform(rbnw)
            dl = self.transform(dl)
            dm = self.transform(dm)
            dr = self.transform(dr)

        lbnw = torch.tensor(lbnw).float().unsqueeze(0)
        mbnw = torch.tensor(mbnw).float().unsqueeze(0)
        rbnw = torch.tensor(rbnw).float().unsqueeze(0)
        dl = torch.tensor(dl).float().unsqueeze(0)
        dm = torch.tensor(dm).float().unsqueeze(0)
        dr = torch.tensor(dr).float().unsqueeze(0)

        return lbnw,mbnw,rbnw,dl,dm,dr

    def _filter_dataset(self,fn):
        """
        Remove the sample corrupted / doesn't have 6 images
        """
        paths = list(self._get_fn(fn))
        exist_fn = list(map(lambda x: os.path.exists(x),paths))
        return all(exist_fn)
