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
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from fusion_funct_multi_3 import score_gen,avg_fun,max_conf_fun,prod_fun,svm_fun,gmean_fun
from keras.utils.np_utils import to_categorical  
from readPreprocess import read_and_preprocess
#Input variable
base='isic'
dataset_name='TEST'
cnn_name='vgg16_19' 
method='svm_fun'
test_path = 'D:/PROJET/ISBI_ARTICLE/base/BASE_MULTI/' + dataset_name
image_size = [224,224]
resultat_final_path='D:/PROJET/REVUE_VISUAL_SENSOR/resultat_final/multi_fusion/' + dataset_name + '/'
name_result_file=dataset_name+'_result_final_'+base+'_cnn_multi_fused_'+method+cnn_name+'.csv'
test_set = glob(test_path + '/*/*.jp*g')

datanorm = ImageDataGenerator(preprocessing_function = preprocess_input)
data = datanorm.flow_from_directory(test_path, target_size=image_size,interpolation='bicubic', shuffle=False, batch_size=1)
test_filenames = data.filenames
test_pathnames=data.filepaths
y_test=data.labels

path=pd.DataFrame(test_filenames,columns=['path'])
filenames=pd.DataFrame(test_filenames,columns=['filenames'])
label=pd.DataFrame(y_test,columns=['label'])
image_size = [224,224] 

test_path = 'D:/PROJET/DICTA_ARTICLE/base/TEST'
[X_test,y_test,test_filenames]=read_and_preprocess(test_path ,len(test_set),preprocess_input,image_size)


#vgg16
cnn_name='vgg16' 
model_name='vgg16'
path_model='D:/PROJET/DICTA_ARTICLE/code/modeles/'+cnn_name+'/'  
path_weigths='D:/PROJET/DICTA_ARTICLE/code/best_weights/'+cnn_name+'/step3/' 
weights_name='vgg16_saved_weights_idx2_epoch12_step3_layer64_lr0.0001.hdf5'
resultat_cnn_path='D:/PROJET/DICTA_ARTICLE/code/resultat/'+cnn_name+'/'
#Load model and weighhts
vgg16 = load_model(path_model+model_name+'.h5')  
vgg16.load_weights(path_weigths+weights_name)
y_test_score_multi = score_gen(len(test_set),test_pathnames,vgg16)
vgg16_score=pd.DataFrame(y_test_score_multi,columns=['bek_16','mel_16','nev_16'])

#vgg19
cnn_name='vgg19' 
model_name='vgg19'
path_model='D:/PROJET/DICTA_ARTICLE/code/modeles/vgg16/'
#path_model='D:/PROJET/DICTA_ARTICLE/code/modeles/'+cnn_name+'/'  
path_weigths='D:/PROJET/DICTA_ARTICLE/code/best_weights/'+cnn_name+'/step3/' 
weights_name='vgg19_saved_weights_idx7_epoch25_step3_layer4_lr0.0001.hdf5'
resultat_cnn_path='D:/PROJET/DICTA_ARTICLE/code/resultat/'+cnn_name+'/'
#Load model and weighhts
vgg19 = load_model(path_model+'vgg16.h5')  
#vgg19 = load_model(path_model+model_name+'.h5')  
vgg19.load_weights(path_weigths+weights_name)
y_test_score_multi = score_gen(len(test_set),test_pathnames,vgg19)
vgg19_score=pd.DataFrame(y_test_score_multi,columns=['bek_19','mel_19','nev_19'])

#resnet50
cnn_name='resnet50' 
model_name='resnet50'
path_model='D:/PROJET/DICTA_ARTICLE/code/modeles/' 
path_weigths='D:/PROJET/DICTA_ARTICLE/code/best_weights/SPLIT_2/'+cnn_name+'/step2/' 
weights_name='resnet50_best_weights_idx12_step2.hdf5'
#Load model and weighhts
resnet50 = load_model(path_model+model_name+'.h5')  
resnet50.load_weights(path_weigths+weights_name)
y_test_score_multi = score_gen(len(test_set),test_pathnames,resnet50)
resnet50_score=pd.DataFrame(y_test_score_multi,columns=['bek_50','mel_50','nev_50'])

result=pd.concat([path,filenames,label,vgg16_score,vgg19_score,resnet50_score],axis=1)
#Load test images

score_bek =np.zeros(len(test_set))
score_mel=np.zeros(len(test_set))
score_nev=np.zeros(len(test_set))
predict_final=np.zeros(len(test_set))

for i in range(len(test_set)):
    bek16=result.bek_16[i] 
    bek19=result.bek_19[i] 
    bek50=result.bek_50[i]
    mel16=result.mel_16[i] 
    mel19=result.mel_19[i] 
    mel50=result.mel_50[i]
    nev16=result.nev_16[i] 
    nev19=result.nev_19[i] 
    nev50=result.nev_50[i]
    #[score_bek[i],score_mel[i],score_nev[i],predict_final[i]]=avg_fun(bek16,bek19,mel16,mel19,nev16,nev19)       
    #[score_bek[i],score_mel[i],score_nev[i],predict_final[i]]=svm_fun(bek16,bek19,mel16,mel19,nev16,nev19) 
    [score_bek[i],score_mel[i],score_nev[i],predict_final[i]]=svm_fun(bek16,bek19,bek50,mel16,mel19,mel50,nev16,nev19,nev50) 
#max_conf_fun,prod_fun,svm_fun,gmean_fun
score_bek_pd=pd.DataFrame(score_bek,columns=['score_bek'])
score_mel_pd=pd.DataFrame(score_mel,columns=['score_mel'])
score_nev_pd=pd.DataFrame(score_nev,columns=['score_nev'])
predict_final_pd=pd.DataFrame(predict_final,columns=['predict_final'])
result_final=pd.concat([result,score_bek_pd,score_mel_pd,score_nev_pd,predict_final_pd],axis=1)
#result_final.to_csv(resultat_final_path+name_result_file)