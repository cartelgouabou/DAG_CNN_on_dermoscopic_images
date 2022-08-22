# -*- coding: utf-8 -*-
"""
Created on Fri May 29 02:11:56 2020

@author: arthu
"""
#Plot heatmap


import matplotlib.pyplot as plt
def plot_heatmap(htmap,img,predicted_label,prob,img_path,save_path):
    pos=img_path.find('ISIC')
    filename=img_path[pos-5:]
    img_name=img_path[pos:]
    save=[save_path + 'hm_'+ img_name]
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    axes[0].imshow(img)
    axes[1].imshow(img)
    i = axes[1].imshow(htmap,cmap="jet",alpha=0.5)
    fig.colorbar(i)
    plt.suptitle("Pr(class={}) = {:5.2f} {}".format(
             predicted_label,
             prob,filename))
    plt.savefig(save[0])
    plt.cla()
    plt.close(fig)