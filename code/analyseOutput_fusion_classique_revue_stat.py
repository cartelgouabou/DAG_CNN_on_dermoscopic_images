# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:17:15 2020

@author: arthu
"""
from __future__ import print_function, division

#Contruction du CNN
#L'étape de préparation de données se fait manuellement


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob  #Module permetant de faire une liste de chemin ayant un nom ou une caractéristiques rechercher
from readPreprocess import read_and_preprocess
from keras.applications.resnet50 import preprocess_input
from Analyse_resultat.resultAnalysis import predict_table,get_images_with_sorted_probabilities,visualize_image
from keras.utils.np_utils import to_categorical  

#Input variable
base='isic'
dataset_name='VALID'
cnn_name='vgg16_19' 
method='avg_fun'
image_size = [224,224]
resultat_final_path='D:/PROJET/REVUE_VISUAL_SENSOR/resultat_final/multi_fusion/' + dataset_name + '/'
name_result_file=dataset_name+'_result_final_'+base+'_cnn_multi_fused_'+method+cnn_name+'.csv'

result_final=pd.read_csv(resultat_final_path+name_result_file)
y_test=result_final.label
y_test_predict=result_final.predict_final
y_test_score=pd.concat([result_final.score_bek,result_final.score_mel,result_final.score_nev],axis=1)
y_test_2D= to_categorical(y_test, num_classes=3)
test_filenames=result_final.filenames


# for i in range(len(y_test_predict)):
#     if y_test_predict[i]==3:
#         y_test_predict[i]=1
  

from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, plot_confusion_matrix,f1_score

cm = confusion_matrix(y_test, y_test_predict)
cm = np.array(confusion_matrix(y_test, y_test_predict, labels=[0,1,2]))
confusion = pd.DataFrame(cm, index=['BEK', 'MEL','NEV'],
                         columns=['predict_BEK','predict_MEL','predict_NEV'])
confusion


target_names = ['BEK', 'MEL','NEV']
print(classification_report(y_test, y_test_predict, target_names=target_names))

print('F1_score: %.3f' % f1_score(y_test,y_test_predict,average='weighted'))
print('Balanced_accuracy: %.3f' % balanced_accuracy_score(y_test,y_test_predict))

#Tracer de la courbe ROC

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score,auc,precision_recall_curve
from matplotlib import pyplot

scores=np.vstack((y_test_score.score_bek,y_test_score.score_mel,y_test_score.score_nev))
scores=np.transpose(scores)
# Calcul de l'AUC
auc = roc_auc_score(y_test_2D, scores,average='weighted')
print('AUC: %.3f' % auc)
# Evaluation de la courbe ROC
fpr = dict()
tpr=dict()
thd=dict()
roc_auc=dict()
for i in range(3):
    fpr[i], tpr[i], thd[i] = roc_curve(y_test_2D[:,i], scores[:,i])
    roc_auc[i]=roc_auc_score(y_test_2D[:,i], scores[:,i])
    (fpr[i], tpr[i])
from itertools import cycle    
colors=cycle(['blue','red','green']) #,'red','purple'
target_names = ['BEK', 'MEL','NEV']
for i, color in zip(range(3),colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (AUC = {1:0.2f})'
             ''.format(target_names[i], roc_auc[i]))
# plt.plot([0, 1], [0, 1], linestyle='--', color='white',label='Mean AUC = 0.91')
# plt.plot([0, 1], [0, 1], linestyle='--', color='white',label='BACC = 0.78')
plt.plot([0, 1], [0, 1], linestyle='--')
# Traver la courbe ROC du modèle 
#plt.plot(fpr, tpr, marker='.')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('1-specificity')
plt.ylabel('Sensitivity')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
# Afficher la courbe
plt.show()

# Calcul de l'AUPRC

# # Evaluation de la courbe PRC
# prec = dict()
# reca=dict()
# #thd=dict()
# prc_auc=dict()
# for i in range(3):
#     prec[i], reca[i], _ = precision_recall_curve(y_test_2D[:,i], scores[:,i])
#     prc_auc[i]=auc(reca[i], prec[i])
#     (prec[i], reca[i])
# from itertools import cycle    
# colors=cycle(['brown','red','green']) #,'red','purple'
# target_names = ['BEK', 'MEL','NEV']
# for i, color in zip(range(3),colors):
#     plt.plot(reca[i], prec[i], color=color, lw=2,
#              label='PR curve of class {0} (AUC = {1:0.2f})'
#              ''.format(target_names[i], prc_auc[i]))

# # plot no skill
# plt.plot([0, 1], [0, 1], linestyle='--')
# # Traver la courbe PRC du modèle 
# #plt.plot(fpr, tpr, marker='.')
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# pyplot.xlabel('Recall')
# pyplot.ylabel('Precision')
# pyplot.title('Precision Recall Curve')
# plt.legend(loc="lower right")
# # Afficher la courbe
# plt.show()
y_test_predict_score_max = np.zeros(len(y_test_score))
for i in range(len(y_test_score)):
    ind=np.argmax(y_test_score.iloc[i])
    y_test_predict_score_max[i]=y_test_score.iloc[i][ind]
    

test_filenames_df=pd.DataFrame(test_filenames,columns=['Filenames'])
y_test_df=pd.DataFrame(y_test,columns=['Label'])
y_test_score_df=pd.DataFrame(y_test_predict_score_max,columns=['Predicted_probability'])
result=pd.concat([test_filenames_df,y_test_df,y_test_score_df],axis=1)
result.to_csv(resultat_cnn_path+name_result_file)


# #Analyse
y_test_score_inv=1-y_test_score
y_test_score_2D=np.concatenate((y_test_score_inv,y_test_score),axis=1)
y_test_predict_2D=np.zeros((len(y_test_predict),2))
for i in range(0,len(y_test_predict)-1):
    y_test_predict_2D[i,0]=1-y_test_predict[i]
    y_test_predict_2D[i,1]=y_test_predict[i]

y_test_2D=np.zeros((len(y_test),2))
for i in range(0,len(y_test_predict)-1):
    y_test_2D[i,0]=1-y_test[i]
    y_test_2D[i,1]=y_test[i]
    
number_of_item=12



result_img_name='Most_confident_pred_benign_keratosis.png'
message = 'Most confident prediction of benign keratosis'
prediction_table= predict_table(y_test_predict_score_max,y_test_predict,y_test)
result_images=get_images_with_sorted_probabilities(prediction_table,
                                         True,
                                         0,
                                         number_of_item,
                                         only_false_predictions=False)
visualize_image(result_images,test_path,test_filenames,message,resultat_cnn_path,result_img_name)

result_img_name='Most_confident_pred_melanoma.png'
message = 'Most confident prediction of melanoma'
prediction_table= predict_table(y_test_predict_score_max,y_test_predict,y_test)
result_images=get_images_with_sorted_probabilities(prediction_table,
                                         True,
                                         1,
                                         number_of_item,
                                         only_false_predictions=False)
visualize_image(result_images,test_path,test_filenames,message,resultat_cnn_path,result_img_name)

result_img_name='Most_confident_pred_nevus.png'
message = 'Most confident prediction of nevi'
prediction_table= predict_table(y_test_predict_score_max,y_test_predict,y_test)
result_images=get_images_with_sorted_probabilities(prediction_table,
                                         True,
                                         2,
                                         number_of_item,
                                         only_false_predictions=False)
visualize_image(result_images,test_path,test_filenames,message,resultat_cnn_path,result_img_name)

result_img_name='Wrong_prediction_of_BEK_with_high_confidence.png'
message = 'Wrong prediction of benign keratosis with high confidence'
prediction_table= predict_table(y_test_predict_score_max,y_test_predict,y_test)
result_images=get_images_with_sorted_probabilities(prediction_table,
                                         True,
                                         0,
                                         number_of_item,
                                         only_false_predictions=True)
visualize_image(result_images,test_path,test_filenames,message,resultat_cnn_path,result_img_name)

result_img_name='Wrong_prediction_of_MEL_with_high_confidence.png'
message = 'Wrong prediction of melanoma with high confidence'
prediction_table= predict_table(y_test_predict_score_max,y_test_predict,y_test)
result_images=get_images_with_sorted_probabilities(prediction_table,
                                         True,
                                         1,
                                         number_of_item,
                                         only_false_predictions=True)
visualize_image(result_images,test_path,test_filenames,message,resultat_cnn_path,result_img_name)


result_img_name='Wrong_prediction_of_NEV_with_high_confidence.png'
message = 'Wrong prediction of nevi with high confidence'
prediction_table= predict_table(y_test_predict_score_max,y_test_predict,y_test)
result_images=get_images_with_sorted_probabilities(prediction_table,
                                         True,
                                         2,
                                         number_of_item,
                                         only_false_predictions=True)
visualize_image(result_images,test_path,test_filenames,message,resultat_cnn_path,result_img_name)

result_img_name='Low_confidence_prediction_of_BEK.png'
message = 'Low confidence prediction of BEK'
prediction_table= predict_table(y_test_predict_score_max,y_test_predict,y_test)
result_images=get_images_with_sorted_probabilities(prediction_table,
                                         False,
                                         0,
                                         number_of_item,
                                         only_false_predictions=False)
visualize_image(result_images,test_path,test_filenames,message,resultat_cnn_path,result_img_name)

result_img_name='Low_confidence_prediction_of_MEL.png'
message = 'Low confidence prediction of MEL'
prediction_table= predict_table(y_test_predict_score_max,y_test_predict,y_test)
result_images=get_images_with_sorted_probabilities(prediction_table,
                                         False,
                                         1,
                                         number_of_item,
                                         only_false_predictions=False)
visualize_image(result_images,test_path,test_filenames,message,resultat_cnn_path,result_img_name)

result_img_name='Low_confidence_prediction_of_NEV.png'
message = 'Low confidence prediction of NEV'
prediction_table= predict_table(y_test_predict_score_max,y_test_predict,y_test)
result_images=get_images_with_sorted_probabilities(prediction_table,
                                         False,
                                         2,
                                         number_of_item,
                                         only_false_predictions=False)
visualize_image(result_images,test_path,test_filenames,message,resultat_cnn_path,result_img_name)
