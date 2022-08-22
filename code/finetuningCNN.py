  # -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 18:58:52 2020

@author: arthu
"""

from __future__ import print_function, division
from builtins import range, input
import sys  #???
#Contruction du CNN
#L'étape de préparation de données se fait manuellement


from keras.preprocessing import image
import tensorflow as tf
from tensorflow import set_random_seed
set_random_seed(12)
from numpy.random import seed
seed(12)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob  #Module permetant de faire une liste de chemin ayant un nom ou une caractéristiques rechercher
from readPreprocess import read_and_preprocess


from keras.callbacks import Callback, EarlyStopping
from math import exp
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,f1_score
from keras.utils.np_utils import to_categorical   
model_name='resnet50_melVSnev'
class roc_callback(Callback):
    def __init__(self,validation_data):
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        # Initialize the best as infinity.
        self.rf = validation_data[2]
        self.lr = validation_data[3]
        self.ep = validation_data[4]
        self.idx= validation_data[5]
        self.step = validation_data[6]
        self.path_history = validation_data[7]
        self.path_weights = validation_data[8]
        self.best_previous = validation_data[9]
        self.best_bacc_val = 0
        self.best_roc_val = 0
        self.best_acc_val = 0# np.Inf
        self.best_acc = 0# np.Inf
        self.best_epoch = 0
        self.best_f1_val=0
        
        #self.w = validation_data[5]
    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        seuil_min=0.5
        y_pred_val2 = np.zeros(len(y_pred_val))
        if epoch==0:
            print('START TRIAL N: %d'%(self.idx))
            f=open(self.path_history+'train_history_step%d_'%(self.step)+model_name+'.txt',"a+")
            line=['START OF TRIAL N:%d\r\n'%(self.idx),"Epoch:%d\r\n" %(epoch),"Learning_rate:%s\r\n" %(self.lr),"Finetuning_rate:%s\r\n" %(self.rf)]
            f.writelines(line)
            f.close()
        for i in range(len(y_pred_val)):
            if y_pred_val[i]>seuil_min:
               y_pred_val2[i]=1
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        bacc_val= balanced_accuracy_score(self.y_val, y_pred_val2)
        val_acc=logs.get('val_accuracy')
        acc=logs.get('accuracy')
        f1_val=f1_score(self.y_val, y_pred_val2)
        if np.less(self.best_bacc_val,bacc_val) :
            self.best_acc=acc
            self.best_acc_val = val_acc
            self.best_roc_val = roc_val
            self.best_bacc_val = bacc_val
            self.best_epoch = epoch
            self.best_f1_val=f1_val
            self.model.save_weights(self.path_weights+model_name+"_best_weights_idx%d_step%d.hdf5" %(self.idx,self.step))
            
        if (bacc_val>0.8) & (roc_val>0.85) & (acc-val_acc<0.1) & (val_acc>0.8) & (acc>0.8) & (f1_val>0.8):#(acc>val_acc) & (acc>0.93):
               self.model.save_weights(self.path_weights+model_name+"_saved_weights_idx%d_epoch%d_step%d_layer%d_lr%s.hdf5" %(self.idx,epoch,self.step,self.rf,self.lr))
               print('\model saved weights_epoch{}'.format(epoch))
               f=open(self.path_history+'train_history_step%d_'%(self.step)+model_name+'.txt',"a+")
               line=['SAVED trial N:%d\r\n'%(self.idx),'Best_model saved\r\n',"Epoch:%d\r\n" %(epoch),"Acc:%s\r\n" %(acc*100),"Val_acc:%s\r\n" %(val_acc*100),"roc_acc:%s\r\n" %(roc_val*100), "bacc_val:%s\r\n"%(bacc_val*100),"f1_val:%s\r\n"%(f1_val*100)]
               f.writelines(line)
               f.close()
        elif (acc-val_acc>0.6) | ((acc-val_acc>0.2) & (acc>0.90)) | (val_acc-acc>0.2) | (f1_val<0.3) : #
               self.model.stop_training = True
               print('\Early model training stopped due to overfiting at epoch{}'.format(epoch))
               f=open(self.path_history+'train_history_step%d_'%(self.step)+model_name+'.txt',"a+")
               line=['END OF A TRIAL:%d\r\n'%(self.idx),'Early_stop\r\n',"Epoch:%s\r\n" %(epoch),"Acc:%s\r\n" %(acc*100),"Val_acc:%s\r\n" %(val_acc*100)]
               lines=['BEST ACHIEVE ON TRIAL OF INDEX:%d\r\n'%(self.idx),"Best_Epoch:%s\r\n" %(self.best_epoch),"Best_acc:%s\r\n" %(self.best_acc*100),"Best_acc_val:%s\r\n" %(self.best_acc_val*100),"Best_roc_val:%s\r\n" %(self.best_roc_val*100),"Best_bacc_val:%s\r\n" %(self.best_bacc_val*100),"Best_f1_val:%s\r\n" %(self.best_f1_val*100)]
               f.writelines(line)
               f.writelines(lines)
               f.close()
        elif (epoch > (self.ep-1)/3)  & (self.best_bacc_val<self.best_previous) : #
               self.model.stop_training = True
               print('\Early model training stopped due to not improvement from previous step at epoch{}'.format(epoch))
               f=open(self.path_history+'train_history_step%d_'%(self.step)+model_name+'.txt',"a+")
               line=['END OF A TRIAL:%d\r\n'%(self.idx),'Early_stop\r\n',"Epoch:%s\r\n" %(epoch),"Acc:%s\r\n" %(acc*100),"Val_acc:%s\r\n" %(val_acc*100)]
               lines=['BEST ACHIEVE ON TRIAL OF INDEX:%d\r\n'%(self.idx),"Best_Epoch:%s\r\n" %(self.best_epoch),"Best_acc:%s\r\n" %(self.best_acc*100),"Best_acc_val:%s\r\n" %(self.best_acc_val*100),"Best_roc_val:%s\r\n" %(self.best_roc_val*100),"Best_bacc_val:%s\r\n" %(self.best_bacc_val*100),"Best_f1_val:%s\r\n" %(self.best_f1_val*100)]
               f.writelines(line)
               f.writelines(lines)
               f.close()
        elif (epoch == (self.ep-1)):
               print('\model training stopped at epoch{}'.format(epoch))
               f=open(self.path_history+'train_history_step%d_'%(self.step)+model_name+'.txt',"a+")
               line=['END OF A TRIAL:%d\r\n'%(self.idx),'Num_epoch reach:%d\r\n'%(epoch),"Best_Epoch:%s\r\n" %(self.best_epoch),"Best_acc:%s\r\n" %(self.best_acc*100),"Best_acc_val:%s\r\n" %(self.best_acc_val*100),"Best_roc_val:%s\r\n" %(self.best_roc_val*100),"Best_bacc_val:%s\r\n" %(self.best_bacc_val*100),"Best_f1_val:%s\r\n" %(self.best_f1_val*100)]
               f.writelines(line)
               f.close()
        return
    
    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
    
    
class LearningRateScheduler(Callback):
  def __init__(self,epo):
    super(LearningRateScheduler, self).__init__()
    #self.lr_schedule = schedule
    self.wait = 0
    self.delay = 8
    # Initialize the best as infinity.
    self.best = np.Inf
    self.diff = np.Inf
    self.ep = epo[0]
    self.initLr=epo[1]
    self.idx =epo[2]
      
  def on_epoch_end(self, epoch, logs=None):
      self.wait+=1
      current = logs.get('val_loss')
      val_acc=logs.get('val_accuracy')
      acc=logs.get('accuracy')
      current2=acc-val_acc
      #print('Index: %d Epoch: %d'%(self.idx,epoch))
      if self.wait > self.delay: 
       self.wait = 0
       if np.less(current, self.best) & np.less(current2,self.diff) & (current2>0):
               lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
               scheduled_lr = lr
               self.best =current
               self.diff=current2
               tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
               #print('loss decrease at Epoch %05d and difference decrease from %d to %d.' %(epoch,self.diff,current2)) 
   
       else:
               lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
               #scheduled_lr=lr * 0.9
               scheduled_lr= self.initLr * (1- (epoch / float(self.ep))) #Poly decay
               tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
               print('loss not decreasing at Epoch %d.' %(epoch))
               print('Learning rate decreased on Epoch %d: new Learning rate is %s.' % (epoch,str(round(scheduled_lr,8))))
      else:      
          self.best =current
          self.diff=current2
      return 

class LearningCurve(Callback):
  
 
  def __init__(self,data):
    
    self.idx =data[0]
    self.path_history = data[1]
    self.step = data[2]
    self.hist=[]
          
  def on_epoch_end(self, epoch, logs=None):
      import numpy as np
      acc = logs.get('accuracy')
      val_acc=logs.get('val_accuracy')
      loss_train= logs.get('loss')
      val_loss = logs.get('val_loss')
      self.hist=np.append(self.hist,[epoch,acc*100,val_acc*100,loss_train,val_loss])
      history=np.reshape(self.hist,(int(len(self.hist)/5),5))
      np.savetxt(self.path_history+'loss_acc_history_step%d_idx%d_'%(self.step,self.idx)+model_name+'.csv',history,header='epoch,acc,val_acc,loss_train,val_loss',delimiter=',',fmt='%.2f')
      if epoch==0:
            f=open(self.path_history+'loss_acc_history_step%d_idx%d_'%(self.step,self.idx)+model_name+'.txt',"a+")
            line=["Epoch;","Acc;","val_acc;","Loss_train;","Loss_test;\n"]
            f.writelines(line)
            f.close()
            f=open(self.path_history+'loss_acc_history_step%d_idx%d_'%(self.step,self.idx)+model_name+'.txt',"a+")
            line=["%d;" %(epoch),"%s;" %(acc*100),"%s;" %(val_acc*100),"%s;" %(loss_train),"%s;\n" %(val_loss)]
            f.writelines(line)
            f.close()
      else :
            f=open(self.path_history+'loss_acc_history_step%d_idx%d_'%(self.step,self.idx)+model_name+'.txt',"a+")
            line=["%d;" %(epoch),"%s;" %(acc*100),"%s;" %(val_acc*100),"%s;" %(loss_train),"%s;\n" %(val_loss)]
            f.writelines(line)
            f.close()
      return 
  
#BASE DE DONNEES

#Importation et préparation des données
#Direction de la base de données

train_path ='D:/PROJET/REVUE_VISUAL_SENSOR/base/melVSnev/TRAIN_GEN'
valid_path ='D:/PROJET/REVUE_VISUAL_SENSOR/base/melVSnev/VALID'

# évalue le nombre d'image dans chaque dossier

train_set = glob(train_path + '/*/*.jp*g')
valid_set = glob(valid_path + '/*/*.jp*g')
# évalue le nombre de classes

numClass = glob(train_path + '/*')

image_size = [224,224]
#image_size = [299,299]
batch_size = 32





from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
#from keras.applications.vgg16 import preprocess_input

    #873
[X_valid2,y_valid2,train_filenames]=read_and_preprocess(valid_path,len(valid_set),preprocess_input,image_size)
train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,      #permettre de normaliser nos données et de mettre toutes les valeurs de nos pixels de [0-255] à [0-1]
                                   # rotation_range=45,
                                   # width_shift_range=0.2,
                                   # height_shift_range=0.2,
                                   # shear_range=0.7,
                                   # zoom_range=0.2,
                                   # horizontal_flip=True,
                                   # vertical_flip=True,
                                   ) 

valid_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)        #pour le jeu de test, permet de normaliser

#les fonctions suivantes sont celles qui vont véritablement agir sur nos images de base et créer les nouvelles images 
train_gen = train_datagen.flow_from_directory(train_path,
                                                 target_size = image_size,   #taille des nouvelles images
                                                 shuffle=True, #Permet de mélanger aléatoirement le jeu de donnée permettant ainsi d'améliorer la qualité du modèle et ainsi que ses performances
                                                 batch_size = batch_size,          #mise à jout des poids apres des lots d'observation
                                                 class_mode = 'binary')

valid_gen = valid_datagen.flow_from_directory(valid_path,
                                            target_size = image_size,
                                            shuffle=False,
                                            batch_size = batch_size,
                                            class_mode = 'binary')

#CONSTRUCTION DU MODELE CNN
# Importation des libraries
from keras.optimizers import Adam

from model_gen_funct import model_gen
step=1
# ratio_freeze=[4,8,16,32,64] 
# learning_rate=[0.00001]
ratio_freeze=[4,8,16,32,64] 
learning_rate=[0.1,0.01,0.001,0.0001,0.00001]
count=0
path_model='D:/PROJET/REVUE_VISUAL_SENSOR/code/modeles/'
path_weights_save='D:/PROJET/REVUE_VISUAL_SENSOR/code/best_weights/melVSnev/step1/'
path_history='D:/PROJET/REVUE_VISUAL_SENSOR/code/training_history/melVSnev/step1/'
#model_name='resnet50'
#weight_init=[0,1]
load_w=False
path_weights='D:/PROJET/REVUE_VISUAL_SENSOR/code/best_weights/melVSnev/step1/'
#weights_filename='resnet50_melVSrest_best_weights_idx14_step1.hdf5'
weights_filename='resnet50_melVSnev_best_weights_idx9_step1.hdf5'

path_weights_load=path_weights+weights_filename
best_previous_bacc_val= 70 #inf if step=1
# tell the model what cost and optimization method to use
for lr in learning_rate:
        for rf in ratio_freeze:
            model=model_gen(rf,1,path_weights_load,load_w)
            count+=1
            if count==1 & step==1:
                model.save(path_model+model_name+'.h5')
            opt=Adam(lr=lr)
            model.compile(
                loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy']
                )
            #ENTRAINEMENT DU MODELE
            # training config:
            epochs =150
            nmel= 779
            nnev = 4694
            nunk= 1538
            wmel = (nmel+nnev+nunk)/nmel
            wnev = (nmel+nnev+nunk)/nnev
            wunk = (nmel+nnev+nunk)/nunk
    
            resultat=model.fit_generator(train_gen,
                             steps_per_epoch=len(train_set)//batch_size,
                             epochs=epochs,
                             verbose=0,
                             validation_data= valid_gen,
                             class_weight={0:wmel,1:wnev},
                             workers=3,
                             validation_steps=len(valid_set)//batch_size,
                             callbacks=[LearningRateScheduler(epo=(epochs,lr,count)),
                                        roc_callback(validation_data=(X_valid2,y_valid2,rf,lr,epochs,count,step,path_history,path_weights_save,best_previous_bacc_val)),
                                        LearningCurve(data=(count,path_history,step))]
                             )
            
    
# for lr in learning_rate:
#         for rf in ratio_freeze:
#             count+=1
#             print(count)
# #model.load_weights('weights_epoch39_step2_layerall_lr00001.hdf5')
            



# model.load_weights("weights_epoch9_step1_layer25_lr001.hdf5") 
# model.save_weights("weights_epoch15_step1_layer3_lr001.hdf5")



# model = load_model('model.h5')

# #EVALUATION DES PERFORMANCE


# # Courbes d'apprentissage

# def courbe_apprentissage(history):
#     print(history.history.keys())
#     # Historique des précisions
#     plt.plot(history.history['accuracy'])  
#     plt.plot(history.history['val_accuracy'])
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
#     # Historique des erreurs
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()

# 	#plt.plot(history.history['val_loss'], color='orange', label='test')
# 	# save plot to file
#     #  filename = sys.argv[0].split('/')[-1]
#     #pyplot.savefig(filename + '_plot.png')
#     #pyplot.close()
    
# courbe_apprentissage(resultat)

# [X_test,y_test,test_filenames]=read_and_preprocess(ps_valid_path ,166,preprocess_input,image_size)
# X_test=X_valid2
# y_test=y_valid2
# [X_test,y_test,test_filenames]=read_and_preprocess(test_path ,1554,preprocess_input,image_size)
# y_test_predict_proba=model.predict(X_test)

# seuil_min=0
# y_test_predict = np.zeros(len(y_test_predict_proba))
# for i in range(len(y_test_predict_proba)):
#     if y_test_predict_proba[i]>seuil_min:
#         y_test_predict[i]=1


# from sklearn.metrics import confusion_matrix, classification_report
# cm = confusion_matrix(y_test, y_test_predict)
# cm = np.array(confusion_matrix(y_test, y_test_predict, labels=[1,0]))
# confusion = pd.DataFrame(cm, index=['Maligne', 'Benigne'],
#                          columns=['predit_maligne','predit_benigne'])
# confusion


# target_names = ['Benigne', 'Maligne']
# print(classification_report(y_test, y_test_predict, target_names=target_names))


# #Tracer de la courbe ROC

# scores= y_test_predict_proba    #prédit le score de confiance des prédiction,il s'agit de la distance entre le vecteur et l'hyperplan
# from sklearn.metrics import roc_curve
# from sklearn.metrics import roc_auc_score
# from matplotlib import pyplot

# # Calcul de l'AUC


# auc = roc_auc_score(y_test, scores)
# print('AUC: %.3f' % auc)
# # Evaluation de la courbe ROC
# fpr, tpr, thresholds = roc_curve(y_test, scores)
# # plot no skill
# pyplot.plot([0, 1], [0, 1], linestyle='--')
# # Traver la courbe ROC du modèle 
# pyplot.plot(fpr, tpr, marker='.')
# pyplot.xlim([-0.05, 1.05])
# pyplot.ylim([-0.05, 1.05])
# pyplot.xlabel('1-spécificité')
# pyplot.ylabel('Sensitivité')
# pyplot.title('Courbe ROC')
# pyplot.legend(loc="lower right")
# # Afficher la courbe
# pyplot.show()


