# -*- coding: utf-8 -*-
"""
Created on Thu May 21 00:05:44 2020

@author: arthu
"""


from __future__ import print_function, division
from builtins import range, input
import sys  #???
#Contruction du CNN
#L'étape de préparation de données se fait manuellement


from keras.preprocessing import image
import tensorflow as tf


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob  #Module permetant de faire une liste de chemin ayant un nom ou une caractéristiques rechercher
from readPreprocess import read_and_preprocess


from keras.models import Model,load_model


path_model='D:/PROJET/DERMA_ARTICLE/code/save/MODELE_NORM/'
path_weight='D:/PROJET/DERMA_ARTICLE/code/save/MODELE_NORM/'    
modelCNN = load_model(path_model+'modelmelVSnevNorm.h5')  #1 2 3 5
modelCNN.load_weights(path_weight+'resnet50_best_weights_idx7_step1_NORM.hdf5')
modelCNN.summary()

modelCNN=Model(inputs=modelCNN.input,outputs=modelCNN.get_layer('avg_pool').output)


from keras.applications.resnet50 import preprocess_input
image_size = [224,224]
train_path = 'D:/PROJET/DERMA_ARTICLE/base/melVSnev/NORM/TRAIN_VALIDATION'
test_path = 'D:/PROJET/DERMA_ARTICLE/base/melVSnev/NORM/TEST'
train_set = glob(train_path + '/*/*.jp*g')
test_set = glob(test_path + '/*/*.jp*g')
[X_train_feat,y_train,y_train_filenames]=read_and_preprocess(train_path ,len(train_set),preprocess_input,image_size)
[X_test_feat,y_test,y_test_filenames]=read_and_preprocess(test_path ,len(test_set),preprocess_input,image_size)

#Extraction des features

X_train=modelCNN.predict(X_train_feat)
X_test=modelCNN.predict(X_test_feat)

import autokeras as ak

clf = ak.StructuredDataClassifier(max_trials=10)
clf.fit(X_train, y_train,validation_split=0.2,epochs=5)

# Predict with the best model.
predicted_y = clf.predict(X_test_feat)
print(predicted_y)


# Evaluate the best model with testing data.
print(clf.evaluate(X_test_feat, y_test))
results = clf.predict(x_test)

from  sklearn.preprocessing import normalize

X_train_norm = normalize(X_train, norm = 'max')
X_test_norm=normalize(X_test,norm='max')

npos=np.sum(y_train==1)
nneg = np.size(y_train) - npos
wpos = nneg/npos 


#importer la librairie ou modele svm

from sklearn.svm import LinearSVC
svc_model = LinearSVC(C=0.001, penalty='l2', max_iter = 6000,class_weight={1:wpos}) 



from sklearn.model_selection import GridSearchCV

parameters = {'C': [0.001,0.01,0.1] }
grid_search = GridSearchCV(estimator = svc_model,
                           param_grid = parameters,
                           scoring = 'balanced_accuracy',
                           cv = 5)
grid_search = grid_search.fit(X_train_norm, y_train)
best_parameters = grid_search.best_params_ 
best_accuracy = grid_search.best_score_ 

#Best model
svc_model = LinearSVC(C=0.1, penalty='l2', max_iter = 5000,class_weight={1:wpos}) 


#Validation du modèle et évaluation des performances à l'aide d'un 20-fold

from sklearn.metrics import make_scorer,recall_score
from sklearn.model_selection import cross_validate

cv_result = cross_validate(estimator =svc_model, X = X_train_norm, y = y_train, cv = 5, 
                           scoring =['accuracy','recall','roc_auc','balanced_accuracy'])
precision = cv_result['test_accuracy'].mean()
precisionStd = cv_result['test_accuracy'].std()
sensibilite = cv_result['test_recall'].mean()
sensibiliteStd = cv_result['test_recall'].std()
auc = cv_result['test_roc_auc'].mean()
aucStd = cv_result['test_roc_auc'].std()
balancedAccuracy = cv_result['test_balanced_accuracy'].mean()
balancedAccuracyStd = cv_result['test_balanced_accuracy'].std()

sens = make_scorer(recall_score,pos_label=1)
spec = make_scorer(recall_score,pos_label=0)

cv_result1 = cross_validate(estimator =svc_model, X = X_train_norm, y = y_train, cv = 5, 
                           scoring = sens)
sensibilite = cv_result1['test_score'].mean()
sensibiliteStd = cv_result1['test_score'].std()

cv_result2 = cross_validate(estimator =svc_model, X = X_train_norm, y = y_train, cv = 5, 
                           scoring = spec)
specificite = cv_result2['test_score'].mean()
specificiteStd = cv_result2['test_score'].std()


svc_model.fit(X_train_norm, y_train)
y_test_predict=svc_model.predict(X_test_norm)
y_test_score=svc_model.decision_function(X_test_norm)


seuil_min=0
y_test_predict = np.zeros(len(y_test_score))
for i in range(len(y_test_score)):
    if y_test_score[i]>seuil_min:
        y_test_predict[i]=1


from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, plot_confusion_matrix

cm = confusion_matrix(y_test, y_test_predict)
cm = np.array(confusion_matrix(y_test, y_test_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['Maligne', 'Benigne'],
                         columns=['predit_maligne','predit_benigne'])
confusion


target_names = ['Benigne', 'Maligne']
print(classification_report(y_test, y_test_predict, target_names=target_names))


print('Balanced_accuracy: %.3f' % balanced_accuracy_score(y_test,y_test_predict))

#Tracer de la courbe ROC

scores= y_test_score    #prédit le score de confiance des prédiction,il s'agit de la distance entre le vecteur et l'hyperplan
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

# Calcul de l'AUC


auc = roc_auc_score(y_test, scores)
print('AUC: %.3f' % auc)


# Evaluation de la courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, scores)
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# Traver la courbe ROC du modèle 
pyplot.plot(fpr, tpr, marker='.')
pyplot.xlim([-0.05, 1.05])
pyplot.ylim([-0.05, 1.05])
pyplot.xlabel('1-spécificité')
pyplot.ylabel('Sensitivité')
pyplot.title('Courbe ROC')
pyplot.legend(loc="lower right")
# Afficher la courbe
pyplot.show()
from joblib import dump,load
dump(svc_model,path_model+'classifierSVMmelVSnev_norm.joblib')