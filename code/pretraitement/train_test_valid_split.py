# # Creating Train / Val / Test folders (One time use)
import os
import numpy as np
import shutil
import random

root_dir = 'D:/PROJET/REVUE_VISUAL_SENSOR/base/BASE_MULTI/ISIC2017/' # data root path
root_source = 'D:/PROJET/REVUE_VISUAL_SENSOR/base/BASE_MULTI/ISIC2017/ISIC_o/' # data root path
classes_dir = ['BEK', 'MEL','NEV'] #total labels 

train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2


for cls in classes_dir:
    os.makedirs(root_dir +'TRAIN/' + cls)
    os.makedirs(root_dir +'VALID/' + cls)
    os.makedirs(root_dir +'TEST/' + cls)
    
    # Creating partitions of the data after shuffeling
    src = root_source + cls # Folder to copy images from

    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)* train_ratio),int(len(allFileNames)* (train_ratio+val_ratio))]) 
 

    train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
    val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, root_dir +'TRAIN/' + cls)

    for name in val_FileNames:
        shutil.copy(name, root_dir +'VALID/' + cls)

    for name in test_FileNames:
        shutil.copy(name, root_dir +'TEST/' + cls)