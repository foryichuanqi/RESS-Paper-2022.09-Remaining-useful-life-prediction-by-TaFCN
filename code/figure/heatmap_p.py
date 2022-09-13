# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 14:37:22 2021

@author: Administrator
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt







#############10   indicators with F1
x  =np. array([[1, 0.20757844233562428, 0.00390625, 0.001953125, 0.001953125, 0.037109375, 1.0, 0.001953125], [0.20757844233562428, 1, 0.001953125, 0.001953125, 0.001953125, 0.00390625, 0.130859375, 0.00390625], [0.00390625, 0.001953125, 1, 0.921875, 0.431640625, 0.275390625, 0.001953125, 0.845703125], [0.001953125, 0.001953125, 0.921875, 1, 0.037109375, 0.921875, 0.001953125, 0.232421875], [0.001953125, 0.001953125, 0.431640625, 0.037109375, 1, 0.16015625, 0.064453125, 0.10546875], [0.037109375, 0.00390625, 0.275390625, 0.921875, 0.16015625, 1, 0.001953125, 0.76953125], [1.0, 0.130859375, 0.001953125, 0.001953125, 0.064453125, 0.001953125, 1, 0.083984375], [0.001953125, 0.00390625, 0.845703125, 0.232421875, 0.10546875, 0.76953125, 0.083984375, 1]])

fig = sns.heatmap(x, annot = True,xticklabels=['LBFS+TaNet','F0+TaNet','F1+TaNet','F1 only','LBFS+A0','LBFS+A1', 'LBFS+A2','LBFS only'],yticklabels=['LBFS+TaNet','F0+TaNet','F1+TaNet','F1 only','LBFS+A0','LBFS+A1', 'LBFS+A2','LBFS only']).invert_yaxis()

plt.xticks(range(0,8),['LBFS+TaNet','F0+TaNet','F1+TaNet','F1 only','LBFS+A0','LBFS+A1', 'LBFS+A2','LBFS only'],rotation=60)
plt.savefig(r'..\..\figure\by_kernel\heatmap_p_.eps',dpi=800,format='eps',bbox_inches = 'tight')

plt.savefig(r'..\..\figure\by_kernel\heatmap_p_.png',dpi=800,format='png',bbox_inches = 'tight')



plt.show()









