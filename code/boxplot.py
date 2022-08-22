# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:35:39 2020

@author: arthu
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#tips = sns.load_dataset("tips")
data=pd.read_csv('test_step1_step2_table.csv')

data.head()

# box=sns.boxplot(x='Lesion',y='Probability',data=data,palette=["blue", "red",'green'],dodge=True)



box=sns.boxplot(x='Lesion',y='Probability',hue='Type',data=data,palette=["brown", 'green'],dodge=True)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);