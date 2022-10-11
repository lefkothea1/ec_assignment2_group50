# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:44:04 2022

@author: lefko
boxplots only-ass2

combine _final files per group (classic-g3 k addvance-g3 sena file) k plot for each group ksexwrisa, clas k adv.
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# setting up

#--------------------------------------------------------------TO PLOT
dir1 = 'D:\\arxeia\\AI_VU\\EvolutionaryComputing\\ass2\\all_exp_data_ass2'
os.chdir(dir1)

# res_box = 'experiment_classic_group2_final.csv'#prob need to combine
res_box = 'experiment_group3_final_allEAs.csv'#prob need to combine

#------reading in REAL results, classic2
df = pd.read_csv(res_box, index_col=(0)) #not skipping any rows


# mybox = sns.boxplot(y='Gain', data=box).set(title='classic_group3')#can already plot boxes by themselves
# plt.savefig('box_classic_g3.png')

# #can already be plotted by the group by, but not nicelly
# df.groupby(df['EA']).boxplot(column='Gain', by='EA') #subplots=False

# # df2 = df.loc[df['enemy']==8] #to plot them separately

# # mybox = sns.boxplot(x='enemy', y='gain', data=df2)#.set(title='base')
mybox = sns.boxplot(x='EA', y='Gain', data=df).set(title='enemy group 3')#-----------------------per enemy group
# # mybox = sns.boxplot(x='enemy', y='gain', data=df).set(title='immigrants')
# # ax.set_xticklabels(["First box","Second box", '3rd']) #can work to customize further
plt.savefig('box_g3_allEAs.png')
# # plt.savefig('box_immigrants.png')




