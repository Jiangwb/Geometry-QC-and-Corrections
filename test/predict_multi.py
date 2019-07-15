# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 15:12:24 2018

@author: Wenbin Jiang
"""
from sklearn.metrics import confusion_matrix
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from matplotlib import pyplot as plt
import scipy.io
from keras.callbacks import EarlyStopping
from keras.models import load_model

# Input:
# image

# Output:
# probability

def loadModel(model_load_path):
    #------------------------Load Model -----------------------------
    # 建立序贯模型
    model = Sequential()
    # returns a compiled model
    # identical to the previous one
    # deletes the existing model
    del model  
    model = load_model(model_load_path)
    model.summary()
    # 配置模型的学习过程
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

#def predprob(data):
#    model = Sequential()
#    # load model
#    model = loadModel('../train/net_all_LMOvel_96traces.h5')
#    model.summary()
#    # 配置模型的学习过程
#    model.compile(loss='categorical_crossentropy',
#                  optimizer='adadelta',
#                  metrics=['accuracy'])
#    
#    if K.image_data_format() == 'channels_first':
#        data = data.reshape(1, 1, img_rows, img_cols)
#    else:
#        data = data.reshape(1, img_rows, img_cols, 1)
#
#    data = data.astype('float32')
#    
#    # predict label  
#    label_pred = model.predict(data)
#    return label_pred

img_rows, img_cols = 64, 96

data = scipy.io.loadmat('./data.mat')
data = data['data']
if K.image_data_format() == 'channels_first':
    data = data.reshape(22801, 1, img_rows, img_cols)
else:
    data = data.reshape(22801, img_rows, img_cols, 1)
data = data.astype('float32')

model = loadModel('../train/net_all_LMOvel_96traces.h5')
label_pred = model.predict(data)
fout = open('prob.txt', 'w')

for i in range(22801):
    probability=label_pred[i-1][0]   # the second row is the probability to be true
    fout.write('%.4f\n' % probability)


