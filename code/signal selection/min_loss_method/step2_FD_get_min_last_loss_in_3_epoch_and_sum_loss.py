
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 20:11:11 2021

@author: Administrator
"""

import os 
import  pandas  as pd 
import numpy as np
def Get_Average(list):

    sum = 0
    
    for item in list:
    
        sum += item
    
    return sum/len(list)  


def get_mp_and_selected_feature(list_last_loss_3epoches,fd_min,shredshold):
    mp=[]
    for i in range(len(list_last_loss_3epoches)):
        mp.append((fd_min-list_last_loss_3epoches[i])/fd_min)
    feature_list=['setting1', 'setting2', 'setting3', 's1_', 's2_', 's3','s4', 's5', 's6', 's7', 's8',
        's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
    selected_feature=[]
    for i in range(len(list_last_loss_3epoches)):
        if mp[i]>shredshold:
            selected_feature.append(feature_list[i])
    
    return mp,selected_feature


shredshold=0          ######### mp_min in Fig.1 of the paper 

def listdir(path, list_name):  #传入存储的list
    for file in os.listdir(path):  
        file_path = os.path.join(path, file)  
        if os.path.isdir(file_path):  
            listdir(file_path, list_name)  
        else:  
            list_name.append(file_path)
list_name=[]
listdir(r'..\..\..\experiments_result\log\feature_select_valid0\fd1',list_name)
feature_columns = ['setting1', 'setting2', 'setting3', 's1_', 's2_', 's3','s4', 's5', 's6', 's7', 's8',
        's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
list_same_feature=[]
list_last_loss_3epoches=[]
list_first_divide_last_loss_3epoches=[]
list_sum_loss_3epoches=[]
for sub_feature in feature_columns:
    for sub_list in list_name:
        if sub_feature in sub_list:
            list_same_feature.append(sub_list)
        list_sum_loss=[]
        list_last_loss=[]
        list_first_loss=[]
        for same_feature in list_same_feature:
            
            df = pd.read_excel(same_feature)  #读取xlsx中第一个sheet

            loss=list(df['root_mean_squared_error']) 
            list_last_loss.append(Get_Average(loss[-10:]))
            list_first_loss.append(loss[0])
            list_sum_loss.append(np.sum(loss))
       
    list_last_loss_3epoches.append(min(list_last_loss))
    # list_first_divide_last_loss_3epoches.append((list_first_loss[list_last_loss.index(min(list_last_loss))]-min(list_last_loss))/list_first_loss[list_last_loss.index(min(list_last_loss))])
    # # print('last{}'.format(min(list_last_loss)))
    # # print('first{}'.format(list_first_loss[list_last_loss.index(min(list_last_loss))]-min(list_last_loss)))
    # # print('map_power{}'.format(list_first_divide_last_loss_3epoches[-1]))
    # list_sum_loss_3epoches.append(list_sum_loss[list_last_loss.index(min(list_last_loss))])
    
    # list_sum_loss=[]
    # list_last_loss=[] 
    # list_first_loss=[]
           
            
            
    list_same_feature=[]

print('FD1_min_average_last_10epoch_in_3experiment')

print(list_last_loss_3epoches)
# print(list_sum_loss_3epoches)
# print(list_first_divide_last_loss_3epoches)

fd1_min=38.50789128965064
fd2_min=38.39994381297593
fd3_min=37.90198469885309
fd4_min=37.37423922740954

mp,selected_feature=get_mp_and_selected_feature(list_last_loss_3epoches,fd1_min,shredshold)
print('mp')

print(mp)


print('selected_feature')
print(selected_feature)















print('\n')










list_name=[]
listdir(r'..\..\..\experiments_result\log\feature_select_valid0\fd2',list_name)
feature_columns = ['setting1', 'setting2', 'setting3', 's1_', 's2_', 's3','s4', 's5', 's6', 's7', 's8',
        's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
list_same_feature=[]
list_last_loss_3epoches=[]
list_first_divide_last_loss_3epoches=[]
list_sum_loss_3epoches=[]
for sub_feature in feature_columns:
    for sub_list in list_name:
        if sub_feature in sub_list:
            list_same_feature.append(sub_list)
        list_sum_loss=[]
        list_last_loss=[]
        list_first_loss=[]
        for same_feature in list_same_feature:
            
            df = pd.read_excel(same_feature)  #读取xlsx中第一个sheet

            loss=list(df['root_mean_squared_error']) 
            list_last_loss.append(Get_Average(loss[-10:]))
            list_first_loss.append(loss[0])
            list_sum_loss.append(np.sum(loss))
       
    list_last_loss_3epoches.append(min(list_last_loss))
    # list_first_divide_last_loss_3epoches.append((list_first_loss[list_last_loss.index(min(list_last_loss))]-min(list_last_loss))/list_first_loss[list_last_loss.index(min(list_last_loss))])
    # # print('last{}'.format(min(list_last_loss)))
    # # print('first{}'.format(list_first_loss[list_last_loss.index(min(list_last_loss))]-min(list_last_loss)))
    # # print('map_power{}'.format(list_first_divide_last_loss_3epoches[-1]))
    # list_sum_loss_3epoches.append(list_sum_loss[list_last_loss.index(min(list_last_loss))])
    
    # list_sum_loss=[]
    # list_last_loss=[] 
    # list_first_loss=[]
           
            
            
    list_same_feature=[]

print('FD2_min_average_last_10epoch_in_3experiment')

print(list_last_loss_3epoches)
# print(list_sum_loss_3epoches)
# print(list_first_divide_last_loss_3epoches)


mp,selected_feature=get_mp_and_selected_feature(list_last_loss_3epoches,fd2_min,shredshold)
print('mp')

print(mp)


print('selected_feature')
print(selected_feature)







print('\n')




list_name=[]
listdir(r'..\..\..\experiments_result\log\feature_select_valid0\fd3',list_name)
feature_columns = ['setting1', 'setting2', 'setting3', 's1_', 's2_', 's3','s4', 's5', 's6', 's7', 's8',
        's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
list_same_feature=[]
list_last_loss_3epoches=[]
list_first_divide_last_loss_3epoches=[]
list_sum_loss_3epoches=[]
for sub_feature in feature_columns:
    for sub_list in list_name:
        if sub_feature in sub_list:
            list_same_feature.append(sub_list)
        list_sum_loss=[]
        list_last_loss=[]
        list_first_loss=[]
        for same_feature in list_same_feature:
            
            df = pd.read_excel(same_feature)  #读取xlsx中第一个sheet

            loss=list(df['root_mean_squared_error']) 
            list_last_loss.append(Get_Average(loss[-10:]))
            list_first_loss.append(loss[0])
            list_sum_loss.append(np.sum(loss))
       
    list_last_loss_3epoches.append(min(list_last_loss))
    # list_first_divide_last_loss_3epoches.append((list_first_loss[list_last_loss.index(min(list_last_loss))]-min(list_last_loss))/list_first_loss[list_last_loss.index(min(list_last_loss))])
    # # print('last{}'.format(min(list_last_loss)))
    # # print('first{}'.format(list_first_loss[list_last_loss.index(min(list_last_loss))]-min(list_last_loss)))
    # # print('map_power{}'.format(list_first_divide_last_loss_3epoches[-1]))
    # list_sum_loss_3epoches.append(list_sum_loss[list_last_loss.index(min(list_last_loss))])
    
    # list_sum_loss=[]
    # list_last_loss=[] 
    # list_first_loss=[]
           
            
            
    list_same_feature=[]

print('FD3_min_average_last_10epoch_in_3experiment')

print(list_last_loss_3epoches)
# print(list_sum_loss_3epoches)
# print(list_first_divide_last_loss_3epoches)


mp,selected_feature=get_mp_and_selected_feature(list_last_loss_3epoches,fd3_min,shredshold)
print('mp')

print(mp)


print('selected_feature')
print(selected_feature)







print('\n')





list_name=[]
listdir(r'..\..\..\experiments_result\log\feature_select_valid0\fd4',list_name)
feature_columns = ['setting1', 'setting2', 'setting3', 's1_', 's2_', 's3','s4', 's5', 's6', 's7', 's8',
        's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
list_same_feature=[]
list_last_loss_3epoches=[]
list_first_divide_last_loss_3epoches=[]
list_sum_loss_3epoches=[]
for sub_feature in feature_columns:
    for sub_list in list_name:
        if sub_feature in sub_list:
            list_same_feature.append(sub_list)
        list_sum_loss=[]
        list_last_loss=[]
        list_first_loss=[]
        for same_feature in list_same_feature:
            
            df = pd.read_excel(same_feature)  #读取xlsx中第一个sheet

            loss=list(df['root_mean_squared_error']) 
            list_last_loss.append(Get_Average(loss[-10:]))
            list_first_loss.append(loss[0])
            list_sum_loss.append(np.sum(loss))
       
    list_last_loss_3epoches.append(min(list_last_loss))
    # list_first_divide_last_loss_3epoches.append((list_first_loss[list_last_loss.index(min(list_last_loss))]-min(list_last_loss))/list_first_loss[list_last_loss.index(min(list_last_loss))])
    # # print('last{}'.format(min(list_last_loss)))
    # # print('first{}'.format(list_first_loss[list_last_loss.index(min(list_last_loss))]-min(list_last_loss)))
    # # print('map_power{}'.format(list_first_divide_last_loss_3epoches[-1]))
    # list_sum_loss_3epoches.append(list_sum_loss[list_last_loss.index(min(list_last_loss))])
    
    # list_sum_loss=[]
    # list_last_loss=[] 
    # list_first_loss=[]
           
            
            
    list_same_feature=[]

print('FD4_min_average_last_10epoch_in_3experiment')

print(list_last_loss_3epoches)
# print(list_sum_loss_3epoches)
# print(list_first_divide_last_loss_3epoches)




mp,selected_feature=get_mp_and_selected_feature(list_last_loss_3epoches,fd4_min,shredshold)
print('mp')

print(mp)


print('selected_feature')
print(selected_feature)





print('\n')





            
            

