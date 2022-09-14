# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 19:44:17 2021

@author: Administrator
"""
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

model_name='1 (9)'
num_test=100

FD='3'



def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true),axis=0))##################  axis=0



    
        
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
#     FD_feature_columns=[ 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's20', 's21']

# if FD=='2':
#     sequence_length=21
#     FD_feature_columns=['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
# if FD=='3':
#     sequence_length=38
#     FD_feature_columns=['s2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's20', 's21']
# if FD=='4':
#     sequence_length=19    
#     FD_feature_columns=['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']



if FD=='1':
    sequence_length=31
    FD_feature_columns=[ 's2', 's3', 's4', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']

if FD=='2':
    sequence_length=21
    FD_feature_columns=['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
if FD=='3':
    sequence_length=38
    FD_feature_columns=[  's2', 's3', 's4', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
if FD=='4':
    sequence_length=19    
    FD_feature_columns=['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's20', 's21']


#############777777777777777777777777777


# if FD=='1':
#     sequence_length=31
#     FD_feature_columns=[ 's1', 's2', 's3','s4', 's5', 's6', 's7', 's8','s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

# if FD=='2':
#     sequence_length=21
#     FD_feature_columns=[ 's1', 's2', 's3','s4', 's5', 's6', 's7', 's8','s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
# if FD=='3':
#     sequence_length=38
#     FD_feature_columns=['s1', 's2', 's3','s4', 's5', 's6', 's7', 's8','s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
# if FD=='4':
#     sequence_length=19    
#     FD_feature_columns=['s1', 's2', 's3','s4', 's5', 's6', 's7', 's8','s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']







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






batch_size=32    
datasets = CMAPSSDataset.CMAPSSDataset(fd_number=FD, batch_size=batch_size, sequence_length=sequence_length,deleted_engine=[1000],feature_columns = FD_feature_columns)#deleted_engine=[5,17,31,41,46,55,69,73,82,95]


train_data = datasets.get_train_data()
train_feature_slice = datasets.get_feature_slice(train_data)
train_label_slice = datasets.get_label_slice(train_data)





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





x_test=np.reshape(test_feature_slice,(-1,test_feature_slice.shape[1],1,test_feature_slice.shape[2]))
test_label_slice[test_label_slice>115]=115
y_test=test_label_slice




print("X_train.shape: {}".format(X_train.shape))
print("Y_train.shape: {}".format(Y_train.shape))

print("X_test.shape: {}".format(x_test.shape))
print("Y_test.shape: {}".format(y_test.shape))
        

model=keras.models.load_model(r"..\..\model\interpretability\{}.h5".format(model_name),custom_objects={'root_mean_squared_error': root_mean_squared_error})

############## Get CAM ################
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages




get_last_conv = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], [model.layers[-4].output])
last_conv = get_last_conv([x_test[:100], 1])[0]

get_softmax = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], [model.layers[-1].output])
softmax = get_softmax(([x_test[:100], 1]))[0]
softmax_weight_4 = model.get_weights()[-4]
softmax_weight_4 =np.reshape(softmax_weight_4,(64,64))


softmax_weight_2 = model.get_weights()[-2]
softmax_weight_2 = np.reshape(softmax_weight_2,(64,1))

softmax_weight=np.matmul(softmax_weight_4, softmax_weight_2)

CAM = np.dot(last_conv, softmax_weight)


sum_x_test=np.sum(x_test.squeeze(),axis=2)
x_test=x_test.squeeze().transpose(0,2,1)



# pp = PdfPages('CAM.pdf')
for k in range(100):
    CAM = (CAM - CAM.min(axis=1, keepdims=True)) / (CAM.max(axis=1, keepdims=True) - CAM.min(axis=1, keepdims=True))
    c = np.exp(CAM) / np.sum(np.exp(CAM), axis=1, keepdims=True)
    # plt.figure(figsize=(13,13));
    # plt.plot(x_test[k].squeeze()[:3]);
    # plt.show()
    # plt.scatter(np.arange(len(x_test[k])), x_test[k].squeeze(), cmap='hot_r', c=c[k, :, :, int(y_test[k])].squeeze(), s=100);
    x=np.arange(0,x_test[k].shape[1])
    # y=x_test[k][1].squeeze()
    # y=np.arange(0,x_test[k].shape[1])
    y=y=sum_x_test[k]
    plt.scatter(x, y, cmap='hot_r', c=c[k, :, 0, 0].squeeze(), s=100);
    plt.plot(x, y, linestyle='--');
    plt.title(
        'True label:' + str(y_test[k][0])[:-2] + '   predicted label ' +': ' + str(softmax[k][0])[:-2])
    plt.colorbar();
    # plt.legend()
    plt.savefig(r'..\..\figure\by_kernel\interpretability\{}_{}.eps'.format(model_name,k),dpi=800,format='eps',bbox_inches = 'tight')
    plt.savefig(r'..\..\figure\by_kernel\interpretability\{}_{}.png'.format(model_name,k),dpi=800,format='png',bbox_inches = 'tight')
    plt.show()




