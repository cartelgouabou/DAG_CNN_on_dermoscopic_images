# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:52:39 2020

@author: arthu
"""



from __future__ import print_function, division
from builtins import range, input
import sys  #???
#Contruction du CNN
#L'étape de préparation de données se fait manuellement


from keras.preprocessing import image



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob  #Module permetant de faire une liste de chemin ayant un nom ou une caractéristiques rechercher

#BASE DE DONNEES

#Importation et préparation des données
#Direction de la base de données

ISIC_path_dest = 'D:/PROJET/REVUE_VISUAL_SENSOR/base/TRAIN_GEN/UNK'
ISIC_path = 'D:/PROJET/REVUE_VISUAL_SENSOR/base/TRAIN/UNK'
# évalue le nombre d'image dans chaque dossier
ISIC_set = glob(ISIC_path + '/*/*.jp*g')


# Affiche une image prise au hasard
plt.imshow(image.load_img(np.random.choice(ISIC_set)))
# Afficher une image précise
#image_name = train_path + '/0/benigneHR1'+'.jpg'
#plt.imshow(image.load_img(image_name))

#Pré-traitement : Génération nouvelles images et resize
image_size = [224,224]
batch_size = 1
#from keras.applications.vgg16 import preprocess_input  #Normalisation appliquer lors de l'entrainement VGG16
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
                              #rescale = 1./255,                            
                              #preprocessing_function=preprocess_input,
                              rotation_range=45,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              #shear_range=0.7,
                              zoom_range=0.2,
                              horizontal_flip=True,
                              vertical_flip=True,
                            )


data = datagen.flow_from_directory(ISIC_path,
                                   target_size=image_size,
                                   shuffle=False, #Permet de mélanger aléatoirement le jeu de donnée permettant ainsi d'améliorer la qualité du modèle et ainsi que ses performances
                                   batch_size=batch_size,
                                   save_to_dir=ISIC_path_dest,
                                   save_prefix='gen',
                                   save_format='jpeg'
                                   #interpolation='bicubic'
                                  )
i=0                                
for x,y in data:
   i+=1
   if i == 962:
       break
  1221 pour 2000 mel
   306 pour 5000 nev
   962 pour 2500 unk