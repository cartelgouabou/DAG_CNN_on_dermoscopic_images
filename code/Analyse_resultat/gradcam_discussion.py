# -*- coding: utf-8 -*-
"""
Created on Wed May 27 04:32:52 2020

@author: arthu
"""

from keras import backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from glob import glob 
from keras.models import Model,load_model

#Load model architecture
path_model='D:/PROJET/ISBI_ARTICLE/code/modeles/'  
model_name='resnet50_bekVSrest.h5'
model = load_model(path_model+model_name) 

# #bekVSaut
# task_name='bekVSrest'
# weights_name='resnet50_bekVSrest_saved_weights_idx3_epoch16_step2_layer16_lr0.1.hdf5'
# step_weigth='step2'
# path_weigths='D:/PROJET/ISBI_ARTICLE/code/best_weights/'+task_name+'/'+ step_weigth + '/'  
# model.load_weights(path_weigths+weights_name)

 
# #melVSaut
# task_name='melVSrest'
# weights_name='resnet50_melVSrest_best_weights_idx8_step3.hdf5'
# step_weigth='step3'
# path_weigths='D:/PROJET/ISBI_ARTICLE/code/best_weights/'+task_name+'/'+ step_weigth + '/'  
# model.load_weights(path_weigths+weights_name)


# # #nevVSaut
# task_name='nevVSrest'
# weights_name='resnet50_nevVSrest_saved_weights_idx24_epoch1_step1_layer64_lr0.0001.hdf5'
# step_weigth='step1'
# path_weigths='D:/PROJET/ISBI_ARTICLE/code/best_weights/'+task_name+'/'+ step_weigth + '/'  
# model.load_weights(path_weigths+weights_name)

# #bekVSmel
# task_name='bekVSmel'
# weights_name='resnet50_bekVSmel_saved_weights_idx7_epoch39_step2_layer16_lr0.01.hdf5'
# step_weigth='step2'
# path_weigths='D:/PROJET/ISBI_ARTICLE/code_step2/best_weights/'+task_name+'/'+ step_weigth + '/'  
# model.load_weights(path_weigths+weights_name)

# #bekVSnev
# task_name='bekVSnev'
# weights_name='resnet50_bekVSnev_saved_weights_idx2_epoch1_step3_layer8_lr0.1.hdf5'
# step_weigth='step3'
# path_weigths='D:/PROJET/ISBI_ARTICLE/code_step2/best_weights/'+task_name+'/'+ step_weigth + '/'  
# model.load_weights(path_weigths+weights_name)



#melVSnev
task_name='melVSnev'
weights_name='resnet50_melVSnev_best_weights_idx9_step5.hdf5'
step_weigth='step5'
path_weigths='D:/PROJET/ISBI_ARTICLE/code_step2/best_weights/'+task_name+'/'+ step_weigth + '/'  
model.load_weights(path_weigths+weights_name)

#modelCNN.summary()
Malign_class = 'Keratosis'
Benign_class = 'Melanoma'

test_path = 'D:/PROJET/ISBI_ARTICLE/base/BASE_MULTI/TEST'
test_path_dest = 'D:/PROJET/ISBI_ARTICLE/resultat_final/discussion/gradcam/'+task_name+ '/' 

test_set = glob(test_path + '/*/*.jp*g')


discussion_analysis_path='D:/PROJET/ISBI_ARTICLE/resultat_final/discussion/'
name_discussion_file='TEST_result_discussion_melanoma_case_111_000_100_110_011.csv'
discussion_analyse=pd.read_csv(discussion_analysis_path+name_discussion_file)
ind=discussion_analyse.ind
ind=ind.drop(labels=[0])
size = len(ind)

for i in [240,252]: #┴mel vs nev 240,252
    #i=242
    img_path=test_set[i]
    img=image.load_img(test_set[i], target_size=(224,224),interpolation='bicubic')
#img=image.load_img(image_path, target_size=(224,224),interpolation='bicubic')
#plt.imshow(img)
    x=image.img_to_array(img)   
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
#print('Predicted:', decode_predictions(preds, top=3)[0])

    np.argmax(preds[0])

    model_output = model.output[:, 0]
    last_conv_layer = model.get_layer('conv5_block3_out')

# Gradients of the Tiger class wrt to the block5_conv3 filer
    grads = K.gradients(model_output, last_conv_layer.output)[0]

# Each entry is the mean intensity of the gradient over a specific feature-map channel 
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

# Accesses the values we just defined given our sample image
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

# Values of pooled_grads_value, conv_layer_output_value given our input image
    pooled_grads_value, conv_layer_output_value = iterate([x])

# We multiply each channel in the feature-map array by the 'importance' 
# of this channel regarding the input image 
    for i in range(2048):
    #channel-wise mean of the resulting feature map is the Heatmap of the CAM
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        heatmap = np.mean(conv_layer_output_value, axis=-1)
 #upsample to 224*224
    # 7*32 = 224

    from plotHeatmap import plot_heatmap
#plot_heatmap(htmap,img,predicted_label,prob)

#upsample size of the grad_cam
    import cv2
    heatmap = cv2.resize(heatmap, (224, 224))
    if preds[0,0]<0.5:
        label_pred=Benign_class
    else:
        label_pred=Malign_class
    score=preds[0,0]
    plot_heatmap(heatmap,img,label_pred,score,img_path,test_path_dest)


#save_img_path = 'C:/Users/arthu/Desktop/nouv_dos/tiger_ca.jpg'
#cv2.imwrite(save_img_path, image)
