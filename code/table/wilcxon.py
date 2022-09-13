# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:12:40 2021

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 11:57:24 2020

@author: Administrator
"""
##########   read datastream from xlsx




##############read  table_data from xlsx
from scipy import stats
import numpy as np
import pandas as pd
path = r'F:\桌面11.17\project\RUL\table\10\RUL_feature_select_comparison_table.xlsx'
# path = r'F:\桌面11.17\project\RUL\table\for_rank1.xlsx'
data = pd.DataFrame(pd.read_excel(path))#读取数据,设置None可以生成一个字典，字典中的key值即为sheet名字，此时不用使用DataFram，会报错


print(data.columns)#获取列的索引名称

rank_list=[]
for i in range(len(data.index)):
    rank_list.append(np.array(data.loc[i].rank()))
rank_array=np.array(rank_list)
average_rank=rank_array.mean(axis=0)
print(average_rank)



################  best rank num  
rank_list=[]
for i in range(len(data.index)):
    rank_list.append(np.array(data.loc[i].rank(method='min')))  ###########choice  min https://blog.csdn.net/weixin_42926612/article/details/90265032
rank_array=np.array(rank_list)

rank_array[rank_array>1]=0
bset_rank=rank_array.sum(axis=0)
print(bset_rank)
bset_rank=bset_rank.sum(axis=0)
print(bset_rank)





    
##################################### T_test    Wilcoxon    









###T_test_ACE
list_triangle=[]
list_triangle_name=[]
for i in range(len(data.columns)):
    list_i=[]
    list_i_name=[]
    for j in range(len(data.columns)):
        list_i.append(stats.ttest_rel(data[data.columns[i]],data[data.columns[j]])[1])
        list_i_name.append([data.columns[i],data.columns[j]])
        
    list_triangle.append(list_i)
    list_triangle_name.append(list_i_name)
print('T_test_ACE:{}'.format(list_triangle))
print('T_test_ACE_name:{}'.format(list_triangle_name))





###wilcoxon_ACE
list_triangle=[]
list_triangle_name=[]
for i in range(len(data.columns)):
    list_i=[]
    list_i_name=[]
    for j in range(len(data.columns)):
        if i==j:
            list_i.append('nan')
        else:
            
            list_i.append(stats.wilcoxon(data[data.columns[i]],data[data.columns[j]])[1])
        list_i_name.append([data.columns[i],data.columns[j]])
        
    list_triangle.append(list_i)
    list_triangle_name.append(list_i_name)
print('wilcoxon_ACE:{}'.format(list_triangle))
print('wilcoxon_ACE_name:{}'.format(list_triangle_name))
        



# T,P =  T_test_PCE(FCN_error,MFCN_error_orignal,num_classes)

# MPCE_FCN=  MPCE(FCN_error,num_classes)
# MPCE_MFCN=  MPCE(MFCN_error_orignal,num_classes)



# print('MACEMACEMACEMACEMACEMACEMACEMACEMACEMACEMACEMACE')

# print('\n')

# print('\n')
# print('\n')
# print('\n')
# print('\n')

# T,P =  T_test_ACE(FCN_error,MFCN_error_orignal,num_classes)
# MACE_FCN=  MACE(FCN_error,num_classes)
# MACE_MFCN=  MACE(MFCN_error_orignal,num_classes)
    
    

# T_test_ACE:[[0.32869356670274663, 0.1515376420321518, 0.09869890530582373, 0.2176672507527079, 0.18651072848790987], [0.18411449988382247, 0.1356641625590175, 0.18394765229446178, 0.08459959452373039], [0.4123627519116271, 0.5073597749739331, 0.26985369803051135], [0.7144507404061191, 0.24312871548995307], [0.2751788402966309], []]
# T_test_ACE_name:[[['R6', 'R7'], ['R6', 'R5'], ['R6', 'W5'], ['R6', 'S6'], ['R6', 'W6']], [['R7', 'R5'], ['R7', 'W5'], ['R7', 'S6'], ['R7', 'W6']], [['R5', 'W5'], ['R5', 'S6'], ['R5', 'W6']], [['W5', 'S6'], ['W5', 'W6']], [['S6', 'W6']], []]
# wilcoxon_ACE:[[0.38818640520827785, 0.0155029296875, 0.00335693359375, 0.0506591796875, 0.02899169921875], [0.083251953125, 0.00762939453125, 0.0506591796875, 0.14385986328125], [0.49542236328125, 0.668548583984375, 0.40374755859375], [0.104583740234375, 0.001312255859375], [0.899932861328125], []]
# wilcoxon_ACE_name:[[['R6', 'R7'], ['R6', 'R5'], ['R6', 'W5'], ['R6', 'S6'], ['R6', 'W6']], [['R7', 'R5'], ['R7', 'W5'], ['R7', 'S6'], ['R7', 'W6']], [['R5', 'W5'], ['R5', 'S6'], ['R5', 'W6']], [['W5', 'S6'], ['W5', 'W6']], [['S6', 'W6']], []]


