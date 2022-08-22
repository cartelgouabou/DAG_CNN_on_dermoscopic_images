# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:17:15 2020

@author: arthu
"""
from __future__ import print_function, division

#Contruction du CNN
#L'étape de préparation de données se fait manuellement




import numpy as np
import pandas as pd
from glob import glob  #Module permetant de faire une liste de chemin ayant un nom ou une caractéristiques rechercher
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from fusion_funct import score_gen,ddag_fun,adag_fun
#Input variable
base='isic'
dataset_name='TEST'
cnn_name='vgg19' 
root_node='02'
test_path = 'D:/PROJET/ISBI_ARTICLE/base/BASE_MULTI/' + dataset_name
image_size = [224,224]
resultat_final_path='D:/PROJET/REVUE_VISUAL_SENSOR/resultat_final/cycle/' + dataset_name + '/'
#name_result_file=dataset_name+'_result_final_'+base+'_cnn_sigmoid_fused_cycle_'+cnn_name+'_'+root_node+'.csv'
name_result_file=dataset_name+'_result_final_'+base+'_cnn_sigmoid_fused_cycle_adag_'+cnn_name+'_'+root_node+'.csv'
test_set = glob(test_path + '/*/*.jp*g')

datanorm = ImageDataGenerator(preprocessing_function = preprocess_input)
data = datanorm.flow_from_directory(test_path, target_size=image_size,interpolation='bicubic', shuffle=False, batch_size=1)
test_filenames = data.filenames
test_pathnames=data.filepaths
y_test=data.labels

path=pd.DataFrame(test_filenames,columns=['path'])
filenames=pd.DataFrame(test_filenames,columns=['filenames'])
label=pd.DataFrame(y_test,columns=['label'])

if cnn_name=='vgg16':
    #VGG16
    #Load model architecture
    path_model='D:/PROJET/ISBI_ARTICLE/code_step2/modeles/'+cnn_name+'/'    
    model_name='vgg16_bekVSmel'
    modelCNN = load_model(path_model+model_name+'.h5')  
    #bekVSmel
    task_name='bekVSmel'
    weights_name='vgg16_bekVSmel_best_weights_idx6_step2.hdf5'
    step_weigth='step2'
    path_weigths='D:/PROJET/ISBI_ARTICLE/code_step2/best_weights/'+cnn_name+'/'+task_name+'/'+ step_weigth + '/'  
    modelCNN.load_weights(path_weigths+weights_name)
    y_test_score_bekVSmel = score_gen(len(test_set),test_pathnames,modelCNN)
    bekVSmel=pd.DataFrame(y_test_score_bekVSmel,columns=[task_name])
    #bekVSnev
    task_name='bekVSnev'
    weights_name='vgg16_bekVSnev_saved_weights_idx6_epoch12_step3_layer32_lr0.0001.hdf5'
    step_weigth='step3'
    path_weigths='D:/PROJET/ISBI_ARTICLE/code_step2/best_weights/'+cnn_name+'/'+task_name+'/'+ step_weigth + '/'  
    modelCNN.load_weights(path_weigths+weights_name)
    y_test_score_bekVSnev = score_gen(len(test_set),test_pathnames,modelCNN)
    bekVSnev=pd.DataFrame(y_test_score_bekVSnev,columns=[task_name])
    #melVSnev
    task_name='melVSnev'
    weights_name='vgg16_melVSnev_best_weights_idx9_step3.hdf5'
    step_weigth='step3'
    path_weigths='D:/PROJET/ISBI_ARTICLE/code_step2/best_weights/'+cnn_name+'/'+task_name+'/'+ step_weigth + '/'  
    modelCNN.load_weights(path_weigths+weights_name)
    y_test_score_melVSnev = score_gen(len(test_set),test_pathnames,modelCNN)
    melVSnev=pd.DataFrame(y_test_score_melVSnev,columns=[task_name])
    result=pd.concat([path,filenames,label,bekVSmel,bekVSnev,melVSnev],axis=1)
elif cnn_name=='vgg19':
    #VGG19
    #Load model architecture
    path_model='D:/PROJET/ISBI_ARTICLE/code_step2/modeles/'+cnn_name+'/'    
    model_name='vgg19_bekVSmel'
    modelCNN = load_model(path_model+model_name+'.h5')  
    #bekVSmel
    task_name='bekVSmel'
    weights_name='vgg19_bekVSmel_best_weights_idx2_step2.hdf5'
    step_weigth='step2'
    path_weigths='D:/PROJET/ISBI_ARTICLE/code_step2/best_weights/'+cnn_name+'/'+task_name+'/'+ step_weigth + '/'  
    modelCNN.load_weights(path_weigths+weights_name)
    y_test_score_bekVSmel = score_gen(len(test_set),test_pathnames,modelCNN)
    bekVSmel=pd.DataFrame(y_test_score_bekVSmel,columns=[task_name])
    #bekVSnev
    task_name='bekVSnev'
    weights_name='vgg19_bekVSnev_saved_weights_idx8_epoch8_step2_layer8_lr0.0001.hdf5'
    step_weigth='step2'
    path_weigths='D:/PROJET/ISBI_ARTICLE/code_step2/best_weights/'+cnn_name+'/'+task_name+'/'+ step_weigth + '/'  
    modelCNN.load_weights(path_weigths+weights_name)
    y_test_score_bekVSnev = score_gen(len(test_set),test_pathnames,modelCNN)
    bekVSnev=pd.DataFrame(y_test_score_bekVSnev,columns=[task_name])
    #melVSnev
    task_name='melVSnev'
    weights_name='vgg19_melVSnev_best_weights_idx5_step2.hdf5'
    step_weigth='step2'
    path_weigths='D:/PROJET/ISBI_ARTICLE/code_step2/best_weights/'+cnn_name+'/'+task_name+'/'+ step_weigth + '/'  
    modelCNN.load_weights(path_weigths+weights_name)
    y_test_score_melVSnev = score_gen(len(test_set),test_pathnames,modelCNN)
    melVSnev=pd.DataFrame(y_test_score_melVSnev,columns=[task_name])
    result=pd.concat([path,filenames,label,bekVSmel,bekVSnev,melVSnev],axis=1)
else:
        print('specifie right cnn architecture')


#0:BEK; 1: MEL; 2: NEV
score_bek =np.zeros(len(test_set))
score_mel=np.zeros(len(test_set))
score_nev=np.zeros(len(test_set))
predict_final=np.zeros(len(test_set))

for i in range(len(test_set)):
    bekVSmel=result.bekVSmel[i] # x>0.5 = 1 ; x<0.5 = 0
    bekVSnev=result.bekVSnev[i] # x>0.5 = 2 ; x<0.5 = 0
    melVSnev=result.melVSnev[i] # x>0.5 = 2 ; x<0.5 = 1
    [score_bek[i],score_mel[i],score_nev[i],predict_final[i]]=ddag_fun(root_node,bekVSmel,bekVSnev,melVSnev)  
    #[score_bek[i],score_mel[i],score_nev[i],predict_final[i]]=adag_fun(root_node,bekVSmel,bekVSnev,melVSnev)           

score_bek_pd=pd.DataFrame(score_bek,columns=['score_bek'])
score_mel_pd=pd.DataFrame(score_mel,columns=['score_mel'])
score_nev_pd=pd.DataFrame(score_nev,columns=['score_nev'])
predict_final_pd=pd.DataFrame(predict_final,columns=['predict_final'])
result_final=pd.concat([result,score_bek_pd,score_mel_pd,score_nev_pd,predict_final_pd],axis=1)
result_final.to_csv(resultat_final_path+name_result_file)