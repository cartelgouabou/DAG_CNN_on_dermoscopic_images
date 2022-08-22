# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:50:00 2020

@author: arthu
"""

def read_and_preprocess(data_path, dataset_size, preprocess_input, image_size):
   import numpy as np
   from keras.preprocessing.image import ImageDataGenerator
   datanorm = ImageDataGenerator(preprocessing_function = preprocess_input)
    # we need to see the data in the same order
    # for both predictions and targets
   data = datanorm.flow_from_directory(data_path, target_size=image_size,interpolation='bicubic', shuffle=False, batch_size=1)
   data_filenames = data.filenames
   X = []
   target = []
   filenames=[]
   i=0
   for x, y in data: 
       i+=1
       s1=x
       #t1=y
       t1=np.argmax(y, axis=1)
       if i==1:
          X=x
          #target=t1
          target=t1
       else :
          X=np.concatenate((X,s1))
          target=np.concatenate((target,t1))
          if i==dataset_size:
              break
   return X,target,data_filenames
