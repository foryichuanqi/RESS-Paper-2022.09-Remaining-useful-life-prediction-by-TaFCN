# -*- coding: utf-8 -*-
"""
Created on Thu May 20 22:06:23 2021

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 18 09:53:38 2021

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 17 09:31:59 2021

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 20:32:47 2021

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 22:23:26 2021

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:26:50 2021

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 11:44:02 2021

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 16:07:41 2021

@author: flc
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:13:20 2021

@author: flc
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 08:05:20 2021

@author: flc
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 21:49:50 2021

@author: flc
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 22:27:27 2021

@author: flc
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:56:51 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 09:09:43 2020

@author: flc
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 11:03:00 2020

@author: flc
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 19:28:39 2020

@author: Administrator
"""

#import tensorflow as tf
import os
import numpy as np
#from numpy import trans
import matplotlib.pyplot as plt
#import tensorflow as tf
import CMAPSSDataset
import pandas as pd
import datetime
import keras
from keras.layers import Lambda
import math
import keras.backend as K
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model



def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true),axis=0))##################  axis=0

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())




    


num_test=100

run_times=10

VALIDATION_SPLIT=0.3

nb_epochs=2000            
batch_size=1024   


patience=50
patience_reduce_lr=20




num_filter1=64
num_filter2=128
num_filter3=64



kernel1_size=16
kernel2_size=10
kernel3_size=6




####31,21,38,19
for FD in['1','2','3','4']: ######['1','2','3','4']

    
    FD_feature_columns=[]
    # if FD=='1':
    #     sequence_length=31
    #     FD_feature_columns=['setting1', 'setting2', 's2', 's3','s4', 's7', 's8','s9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    
    # if FD=='2':
    #     sequence_length=21
    #     FD_feature_columns=['s3','s4','s11','s15','s17']
    # if FD=='3':
    #     sequence_length=38
    #     FD_feature_columns=['setting1', 'setting2',  's2', 's3','s4',  's7', 's8','s9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    # if FD=='4':
    #     sequence_length=19    
    #     FD_feature_columns=['s3','s4','s11','s15','s17']#['s3','s4','s11','s17']
    
    
    
    # if FD=='1':
    #     sequence_length=31
    #     FD_feature_columns=['setting1', 'setting2', 's2', 's3','s4', 's7', 's8','s9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    
    # if FD=='2':
    #     sequence_length=21
    #     FD_feature_columns=['setting1', 'setting2', 's2', 's3','s4', 's7', 's8','s9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    # if FD=='3':
    #     sequence_length=38
    #     FD_feature_columns=['setting1', 'setting2', 's2', 's3','s4', 's7', 's8','s9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    # if FD=='4':
    #     sequence_length=19    
    #     FD_feature_columns=['setting1', 'setting2', 's2', 's3','s4', 's7', 's8','s9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']

    


#####################################333333333333333333333333333333333
    
    # if FD=='1':
    #     sequence_length=31
    #     FD_feature_columns=['setting1', 'setting2', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's20', 's21']

    # if FD=='2':
    #     sequence_length=21
    #     FD_feature_columns=['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
    # if FD=='3':
    #     sequence_length=38
    #     FD_feature_columns=[ 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's20', 's21','setting1', 'setting2']
    # if FD=='4':
    #     sequence_length=19    
    #     FD_feature_columns=['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']




#################################    44444444444444444444444444
    
    # if FD=='1':
    #     sequence_length=31
    #     FD_feature_columns=['setting1', 'setting2', 'setting3', 's1', 's2', 's3','s4', 's5', 's6', 's7', 's8','s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

    # if FD=='2':
    #     sequence_length=21
    #     FD_feature_columns=['setting1', 'setting2', 'setting3', 's1', 's2', 's3','s4', 's5', 's6', 's7', 's8','s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
    # if FD=='3':
    #     sequence_length=38
    #     FD_feature_columns=['setting1', 'setting2', 'setting3', 's1', 's2', 's3','s4', 's5', 's6', 's7', 's8','s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
    # if FD=='4':
    #     sequence_length=19    
    #     FD_feature_columns=['setting1', 'setting2', 'setting3', 's1', 's2', 's3','s4', 's5', 's6', 's7', 's8','s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']


 #############################5555555555555555555555555555   

    
    # if FD=='1':
    #     sequence_length=31
    #     FD_feature_columns=[  's2', 's3','s4',  's7', 's8','s9', 's11', 's12', 's13', 's14', 's15', 's17',  's20', 's21']

    # if FD=='2':
    #     sequence_length=21
    #     FD_feature_columns=[  's2', 's3','s4',  's7', 's8','s9', 's11', 's12', 's13', 's14', 's15', 's17',  's20', 's21']
    # if FD=='3':
    #     sequence_length=38
    #     FD_feature_columns=[  's2', 's3','s4',  's7', 's8','s9', 's11', 's12', 's13', 's14', 's15', 's17',  's20', 's21']
    # if FD=='4':
    #     sequence_length=19    
    #     FD_feature_columns=[  's2', 's3','s4',  's7', 's8','s9', 's11', 's12', 's13', 's14', 's15', 's17',  's20', 's21']
        


#############################666666666666666666666666666666666            
    # if FD=='1':
    #     sequence_length=31
    #     FD_feature_columns=[  's2', 's3','s4',  's7', 's8','s9', 's11', 's12', 's13', 's14', 's15', 's17',  's20', 's21']

    # if FD=='2':
    #     sequence_length=21
    #     FD_feature_columns=['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
    # if FD=='3':
    #     sequence_length=38
    #     FD_feature_columns=[  's2', 's3','s4',  's7', 's8','s9', 's11', 's12', 's13', 's14', 's15', 's17',  's20', 's21']
    # if FD=='4':
    #     sequence_length=19    
    #     FD_feature_columns=['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']


#############777777777777777777777777777


    if FD=='1':
        sequence_length=31
        FD_feature_columns=[ 's1', 's2', 's3','s4', 's5', 's6', 's7', 's8','s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

    if FD=='2':
        sequence_length=21
        FD_feature_columns=[ 's1', 's2', 's3','s4', 's5', 's6', 's7', 's8','s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
    if FD=='3':
        sequence_length=38
        FD_feature_columns=['s1', 's2', 's3','s4', 's5', 's6', 's7', 's8','s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
    if FD=='4':
        sequence_length=19    
        FD_feature_columns=['s1', 's2', 's3','s4', 's5', 's6', 's7', 's8','s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']






    method_name='grid_FD{}_multi_channel_one_exp_attetion_num_test{}'.format(FD,num_test)
    # method_name='FCN_RUL_1out_train_split_test'
    dataset='cmapssd'
    
    
    def unbalanced_penalty_score_1out(Y_test,Y_pred) :
          
        s=0    
        for i in range(len(Y_pred)):
            if Y_pred[i]>Y_test[i]:
                s=s+math.exp((Y_pred[i]-Y_test[i])/10)-1
            else:
                s=s+math.exp((Y_test[i]-Y_pred[i])/13)-1    
        print('unbalanced_penalty_score{}'.format(s))
        return s  
      
    def error_range_1out(Y_test,Y_pred) :           
        error_range=(Y_test-Y_pred).min(),(Y_test-Y_pred).max()
        print('error range{}'.format(error_range))
        return error_range
    

    
    
    # all_feature_columns =['setting1', 'setting2', 'setting3', 's1', 's2', 's3','s4', 's5', 's6', 's7', 's8','s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
    # for i_all_feature_columns in range(len(all_feature_columns)):
        
    datasets = CMAPSSDataset.CMAPSSDataset(fd_number=FD, batch_size=batch_size, sequence_length=sequence_length,deleted_engine=[1000],feature_columns = FD_feature_columns)#deleted_engine=[5,17,31,41,46,55,69,73,82,95]
    
    
    train_data = datasets.get_train_data()
    train_feature_slice = datasets.get_feature_slice(train_data)
    train_label_slice = datasets.get_label_slice(train_data)
    
    # valid_feature_slice = datasets.get_valid_feature_slice(train_data)
    # valid_label_slice = datasets.get_valid_label_slice(train_data)
    
    
    
    print("train_data.shape: {}".format(train_data.shape))
    print("train_feature_slice.shape: {}".format(train_feature_slice.shape))
    print("train_label_slice.shape: {}".format(train_label_slice.shape))
    
    
    test_data = datasets.get_test_data()
    if num_test==100:
        
        test_feature_slice, test_label_slice = datasets.get_last_data_slice(test_data)
        
    if num_test==10000:
        
        test_feature_slice = datasets.get_feature_slice(test_data)
        test_label_slice = datasets.get_label_slice(test_data)
    # test_feature_slice, test_label_slice = datasets.get_last_data_slice(test_data)
    
    
    print("test_data.shape: {}".format(test_data.shape))
    print("test_feature_slice.shape: {}".format(test_feature_slice.shape))
    print("test_label_slice.shape: {}".format(test_label_slice.shape))
    
    timesteps = train_feature_slice.shape[1]
    input_dim = train_feature_slice.shape[2]
    
    #train_feature_slice=np.transpose( train_feature_slice,(0,2,1))
    #
    #
    #test_feature_slice=np.transpose( test_feature_slice,(0,2,1))
    
    
    
    X_train=np.reshape(train_feature_slice,(-1,train_feature_slice.shape[1],1,train_feature_slice.shape[2]))
    train_label_slice[train_label_slice>115]=115
    Y_train=train_label_slice
    
    N=Y_train.shape[0]
    
    
    print('FD{}_N={}'.format(FD,N))
    
    A=0
    for i in range(Y_train.shape[0]):
        A+=Y_train[i][0]
    
    A=-2*A/N  
    
    
    print('FD{}A={}'.format(FD,A))
    B=0
    for i in range(Y_train.shape[0]):
        B+=pow(Y_train[i][0],2)/N
        
    print('FD{}B={}'.format(FD,B))
    
    minloss=((-0.5*A)**2-A**2/2+B)**0.5    ########  loosb in Eq.(1),where x=-0.5*A
    print('FD{}minloss={}'.format(FD,minloss))
    

            
        
            
        




