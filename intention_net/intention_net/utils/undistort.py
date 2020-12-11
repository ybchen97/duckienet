import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt

#rgb_0
FRONT_CAMERA_INFO = { 'K': [714.3076782226562, 0.0, 642.54052734375, 0.0, 714.65966796875, 382.0349426269531, 0.0, 0.0, 1.0],
                'D' : [-0.3165473937988281, 0.1023712158203125, -1.52587890625e-05, -0.000728607177734375, 0.0],
                'R' : [0.9999804496765137, -0.006237626075744629, 0.0003840923309326172, 0.006237149238586426, 0.9999796152114868, 0.001311659812927246, -0.0003923177719116211, -0.0013092756271362305, 0.9999990463256836] ,
                'P' : [698.4000244140625, 0.0, 649.08251953125, 0.0, 0.0, 698.4000244140625, 377.290771484375, 0.0, 0.0, 0.0, 1.0, 0.0]}

#rgb_4
RIGHT_CAMERA_INFO = {
    'K': [365.27876922993573, 0.0, 377.2266370410107, 0.0, 367.43567665349434, 239.7490459357185, 0.0, 0.0, 1.0],
    'D': [-0.021799544117355064, -0.013164107936537923, 0.0319940345464826, -0.02443985962443235,0.0],
    'R': [0.9997644032802752, -0.0008011791434607397, 0.021690920810685264, 0.0007894321192680866, 0.9999995370815493, 0.0005501214556035449, -0.021691351515394412, -0.0005328683392086987, 0.9997645729474356],
    'P': [366.6972022844292, 0.0, 335.4812469482422, 0.0, 0.0, 366.6972022844292, 258.4048194885254, 0.0, 0.0, 0.0, 1.0, 0.0]      
}

#rgb_7
LEFT_CAMERA_INFO = {
    'K': [367.24613248988425, 0.0, 383.58120792452786, 0.0, 367.18992512939246, 246.0435193107078, 0.0, 0.0, 1.0],
    'D': [-0.023424071222704918, 0.0029950533726772295, -0.006144382406253766, 0.0014457766750990203,0.0],
    'R': [0.999983421344173, 0.0004443354710229224, 0.005741045444123884, -0.0004637201933800672, 0.9999941950444479, 0.003375624533184727, -0.005739512207893308, -0.0033782308085709436, 0.9999778225321897],
    'P': [367.1950001882657, 0.0, 367.7484130859375, -44131.81047519387, 0.0, 367.1950001882657, 251.78948974609375, 0.0, 0.0, 0.0, 1.0, 0.0]   
}

def undistort(img,param=FRONT_CAMERA_INFO):
    # K - Intrinsic camera matrix for the raw (distorted) images.
    camera_matrix =  param['K']
    camera_matrix = np.reshape(camera_matrix, (3, 3))

    # distortion parameters - (k1, k2, t1, t2, k3)
    distortion_coefs = param['D']
    distortion_coefs = np.reshape(distortion_coefs, (1, 5))

    # R - Rectification matrix - stereo cameras only, so identity
    rectification_matrix =  param['R']
    rectification_matrix = np.reshape(rectification_matrix,(3,3))
    
    # P - Projection Matrix - specifies the intrinsic (camera) matrix
    #  of the processed (rectified) image
    projection_matrix = param['P']
    projection_matrix = np.reshape(projection_matrix, (3, 4))
    
    # Not initialized - initialize all the transformations we'll need
    mapx = np.zeros(img.shape)
    mapy = np.zeros(img.shape)

    H = img.shape[0]
    W = img.shape[1]

    # Initialize self.mapx and self.mapy (updated)
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, 
        distortion_coefs, rectification_matrix, 
        projection_matrix, (W, H), cv2.CV_32FC1)

    return cv2.remap(img, mapx, mapy, cv2.INTER_NEAREST)
    
if __name__ == '__main__':
    print('hello')
    for img in glob.glob("../test/*.jpg"):
        print(img)
        img = cv2.imread(img)
        img = undistort(img)
        plt.imshow(img)
        plt.show()
    
