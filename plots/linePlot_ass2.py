# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:28:02 2022

@author: lefko
lineplotting ass2 ec
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dir1 = 'D:\\arxeia\\AI_VU\\EvolutionaryComputing\\ass2\\all_exp_data_ass2\\group2'
# dir1 = 'D:\\arxeia\\AI_VU\\EvolutionaryComputing\\ass2\\all_exp_data_ass2'
os.chdir(dir1)

#-----------------loading classic data
df = pd.read_csv('experiment_classic_group2_results.csv', index_col=(0))

# filtering out [and ] char from cols
# d[0] =  d[0].str.strip('[').astype(int)
cols_2fix = ['max', 'avg', 'min', 'std' ]
for col in cols_2fix:
    df[col] = df[col].str.strip('[]').astype(float) 
    
#grouping:preparing data to plot
mean_clas2 = df.groupby('gen')['avg'].mean() 
mean_clas2_std = df.groupby('gen')['avg'].std() 
max_clas2 = df.groupby('gen')['max'].mean() 
max_clas2_std = df.groupby('gen')['max'].std() 

#-----------------loading advanced data 
df2 = pd.read_csv('exp_adv_g2_results_CLEAN.csv', index_col=(0))
    
#grouping FITNESS VALUES:preparing data to plot
mean_adv = df2.groupby('gen')['avg_fit'].mean() 
mean_adv_std = df2.groupby('gen')['avg_fit'].std() 
max_adv = df2.groupby('gen')['max_fit'].mean() 
max_adv_std = df2.groupby('gen')['max_fit'].std() 

a=0.15
#plotting everything
x = np.arange(30)#gen number
ax = plt.subplot(111)
# conv mean n max FOR CLASSIC
ax.plot(x, mean_clas2, label='mean_classic' , color = 'blue')
plt.fill_between(x, mean_clas2-mean_clas2_std, mean_clas2 + mean_clas2_std, alpha=a, color = 'blue') 
ax.plot(x, max_clas2, label='max_classic', color = 'dodgerblue') 
plt.fill_between(x,  max_clas2 - max_clas2_std,  max_clas2 + max_clas2_std,  alpha=a, color = 'deepskyblue')

# conv mean n max FOR ADVANCED
ax.plot(x, mean_adv, label='mean_adv' , color='lightcoral')
plt.fill_between(x, mean_adv-mean_adv_std, mean_adv + mean_adv_std, alpha=a, color = 'lightcoral')
ax.plot(x, max_adv, label='max_adv' , color='red')
plt.fill_between(x,  max_adv - max_adv_std,  max_adv + max_adv_std,  alpha=a, color = 'red')


plt.title('enemy group 2', fontsize=20)
plt.ylabel('mean fitness', fontsize=20)#, labelsize=20)
plt.xlabel('generations', fontsize=20)#, labelsize=20)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('line_group2_allEAs.png')
