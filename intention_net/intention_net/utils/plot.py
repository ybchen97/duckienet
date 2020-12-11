import glob
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.utils import to_categorical
import numpy as np
from dataset import PioneerDataset as Dataset
from net import FeatModel, IntentionNet
feat_model = FeatModel()
net = IntentionNet('DLM','NORMAL', 2, 4)
net.load_weights('/data/model/mae/NORMAL_DLM_latest_model.h5')
train_generator = Dataset('/data', 16, 4, input_frame='NORMAL',mode='DLM')
true_x = []
true_z = []
pred_x = []
pred_z = []
intent = []
print(train_generator.__len__())
for i in range(100):
    data,label = train_generator.__getitem__(i)
    intention = np.array(data[1],dtype='float64')
    w = np.array([0,0,1,-1],dtype='float64').reshape(4,1)
    intention = intention.dot(w)
    pred = net.predict(data)
    true_x = true_x + list(np.array(label,dtype='float64')[:,0])
    true_z = true_z + list(np.array(label,dtype='float64')[:,1])
    pred_x = pred_x + list(np.array(pred,dtype='float64')[:,0])
    pred_z = pred_z + list(np.array(pred,dtype='float64')[:,1])
    intent = intent + list(np.array(intention,dtype='float64').reshape(-1))

import matplotlib.pyplot as plt
plt.plot(np.arange(len(true_z)),true_z,color='blue')
plt.plot(np.arange(len(pred_z)),pred_z,color='red')
#plt.plot(np.arange(len(intent)),intent,color='yellow')

plt.show()