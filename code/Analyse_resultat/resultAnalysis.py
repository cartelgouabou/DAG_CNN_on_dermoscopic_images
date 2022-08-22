# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 10:31:47 2020

 
most_confident_nevus_images = get_images_with_sorted_probabilities(prediction_table, True, 0, 10,
                                               False)
message = 'Images with highest probability of containing nevus'
visualize_image(data_path,data_filenames,most_confident_nevus_images, message)  
@author: arthu
"""



import numpy as np

#y_test_score: probability or score of the classifier; y_test_predict:predicted class,y_test: ground true label
def predict_table(y_test_score_2D,y_test):
    prediction_table = {}
    for i in range(len(y_test_score_2D)):
        ind_of_higher_prob=np.argmax(y_test_score_2D[i])
        value_of_high_prob=y_test_score_2D[i,ind_of_higher_prob]
        label=y_test[i]
        prediction_table[i]=[value_of_high_prob,ind_of_higher_prob,label]
    assert len(y_test_score_2D) == len(y_test) == len(prediction_table)
    return prediction_table

#get_highest_probability(bolean): get the highest probability(true) or the lowest(false)
        #label: predicted class;#number_of_item: number of images to retain
def get_images_with_sorted_probabilities(prediction_table,
                                         get_highest_probability,
                                         label,
                                         number_of_items,
                                         only_false_predictions=False):
        
    #Trier par ordre croissant ou decroissant    
    sorted_prediction_table = [(k, prediction_table[k])
                               for k in sorted(prediction_table,
                                               key=prediction_table.get,
                                               reverse=get_highest_probability)
                               ]
    
    result = []
    for index, key in enumerate(sorted_prediction_table):
        image_index, [probability, predicted_index, gt] = key
        if predicted_index == label:
            if only_false_predictions == True:
                if predicted_index != gt:
                    result.append(
                        [image_index, [probability, predicted_index, gt]])
            else:
                result.append(
                    [image_index, [probability, predicted_index, gt]])
    return result[:number_of_items]



def visualize_image(sorted_indices,data_path,data_filenames,msg,resultat_path,result_img_name):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
      
    def plot_images(filenames, distances, message):
        images = []
        for filename in filenames:
            images.append(mpimg.imread(filename))
            plt.figure(figsize=(20, 20))
            columns = 4
            for i, image in enumerate(images):
                ax = plt.subplot(len(images) / columns + 1, columns, i + 1)
                ax.set_title("\n\n" + filenames[i].split("/")[-1] + "\n" +
                             "\nProbability: " +
                             str(distances[i])) #"{0:.3f}".format(distances[i]))
                plt.suptitle(message, fontsize=20, fontweight='bold')
                plt.axis('off')
                plt.imshow(image)
                plt.savefig(resultat_path+result_img_name) #
                
                #
        
    similar_image_paths = []
    distances = []
    for name, value in sorted_indices:
        [probability, predicted_index, gt] = value
        similar_image_paths.append(data_path +'/' + data_filenames[name])
        distances.append(probability)
    plot_images(similar_image_paths,distances,msg)
          
             

