# -*- coding: utf-8 -*-
"""
Created on Mon May 25 05:39:13 2020

@author: arthu
"""


from __future__ import print_function, division
from builtins import range, input
import sys  #???


from keras.preprocessing import image
import tensorflow as tf


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import scipy as sp


test_path = 'D:/PROJET/DERMA_ARTICLE/base/mednote/TEST_05'
test_set = glob(test_path + '/*/*.jp*g')

test_path = 'D:/PROJET/DERMA_ARTICLE/base/melVSnev/NORM/TEST_45'
test_set = glob(test_path + '/*/*.jp*g')

plt.imshow(image.load_img(np.random.choice(test_set)))
plt.show()

from keras.models import Model,load_model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import preprocess_input, decode_predictions

modelCNN = load_model('modelmelVSnev.h5')
#modelCNN.summary()
modelCNN.load_weights("weights_epoch39_step2_layerall_lr000001.hdf5") 
modelCNN.summary()
#Prepare a model to get output before flatten
modelConv=Model(inputs=modelCNN.input,outputs=modelCNN.get_layer('activation_49').output)

#get feature map weights
final_dense= modelCNN.get_layer('dense_2')
W=final_dense.get_weights()[0]


final_dense2= modelCNN.get_layer('dense_1')
W2=final_dense2.get_weights()[0]
#while True:
    img=image.load_img(np.random.choice(test_set), target_size=(224,224))
    x=preprocess_input(np.expand_dims(img,0))
    fmap=modelConv.predict(x)[0] #7*7*2048
    
    # get predicted class
    #probs = modelCNN.predict(x)
    #classnames='melanome'
    #classnames = decode_predictions(probs)[0]
    #print(classnames)
    #classname= classnames[0][1]
    #pred = np.argmax(probs[0])
    w2=np.mean(W2,axis=1)
    #get the 2048 weigths for the relevant class
    #w=W[:,pred]
    #dot w with fmaps
    #cam = fmap.dot(w)
    cam = fmap.dot(w2)
    #upsample to 224*224
    # 7*32 = 224
    cam= sp.ndimage.zoom(cam,(32,32), order=1)
    
    plt.subplot(1,2,1)
    plt.imshow(img, alpha=0.8)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.subplot(1,2,2)
    plt.imshow(img)