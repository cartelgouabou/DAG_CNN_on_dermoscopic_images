# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 17:41:19 2020

@author: arthu
"""
#ration_freeze :percentage of layer to finetune
#w type of weight initialisation 1 for imagenet and 0 for random
#

def model_gen(ratio_freeze,w,path_weights,load):
    from keras.layers import Input, Dropout, Dense, GlobalAveragePooling2D
    from keras.models import Model,load_model
    #from keras.applications.mobilenet import MobileNet
    #from keras.applications.mobilenet_v2 import MobileNetV2
    #from keras.applications.nasnet import NASNetMobile
    #from keras.applications.nasnet import NASNetLarge
    #from keras.applications.vgg19 import VGG19
    #from keras.applications.efficientnet import EfficientNetB0
    from keras.applications.resnet import ResNet50
    from keras import initializers
    if w == 1:
        model_O=ResNet50(input_shape=(224, 224, 3),weights='imagenet',include_top=True) #'weights=imagenet'
    else:
        model_O=ResNet50(input_shape=(224, 224, 3),include_top=False) #'weights=imagenet'
    num_layer=len(model_O.layers)
    num_freeze_layer= (ratio_freeze*num_layer)//100
    model_O.layers.pop()
    prediction=model_O.layers[-1].output
    #prediction=GlobalAveragePooling2D()(prediction)
    prediction=Dense(1024,kernel_initializer='random_normal',activation='relu')(prediction)
    prediction=Dropout(0.5)(prediction)
    prediction = Dense(1, kernel_initializer='random_normal',activation='sigmoid')(prediction)
    model = Model(inputs=model_O.input, outputs=prediction)
    for layer in model.layers:
        layer.trainable = True

    for layer in model.layers[:-num_freeze_layer]:
        layer.trainable = False   
    if load==True:
        model.load_weights(path_weights) 
    return model
