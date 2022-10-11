# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 12:04:19 2022

@author: lefko
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dir1 = 'D:\\arxeia\\AI_VU\\EvolutionaryComputing\\ass2\\all_exp_data_ass2'
os.chdir(dir1)

df2 = pd.read_csv('experiment_advanced_group3_results.csv', index_col=(0))

# filtering out [and ] char from cols
# d[0] =  d[0].str.strip('[').astype(int)
cols_2fix = ['max', 'avg', 'min', 'std' ]
for col in cols_2fix:
    df2[col] = df2[col].str.strip('[]')
    
# for col in d2.columns:
#     d2[[col,col]] = d2[col].str.split(":",expand=True)
df=df2[['gen', 'nevals']].copy()
df[['max_fit', 'max_win']]=df2['max'].str.split(" ",n=1, expand=True).astype(float)
df[['avg_fit', 'avg_win']]=df2['avg'].str.split(" ",n=1, expand=True).astype(float)
df[['min_fit', 'min_win']]=df2['min'].str.split(" ",n=1, expand=True).astype(float)
df[['std_fit', 'std_win']]=df2['std'].str.split(" ",n=1, expand=True).astype(float)

df.to_csv('exp_adv_g3_results_CLEAN.csv')