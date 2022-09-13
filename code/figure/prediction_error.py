# # -*- coding: utf-8 -*-
# """
# Created on Tue Aug 31 15:36:08 2021

# @author: Administrator
# """

# # -*- coding: utf-8 -*-
# """
# Created on Fri Apr 23 20:19:12 2021

# @author: Administrator
# """

# # -*- coding: utf-8 -*-
# """
# Created on Wed Apr  7 19:44:17 2021

# @author: Administrator




# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 15:36:08 2021

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 20:19:12 2021

@author: Administrator
"""

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
from tfdeterminism import patch
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
import copy 


num_test=10000

id_num=4
# sequence_length=1
# FD_feature_columns=[]

FD='1'

avg_len=1

      
        
dispaly='predict'   
# dispaly='error'               
dispaly='error_abs' 


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
# if method=='LBFS+TaNet':
#     sequence_length=38
#     FD_feature_columns=[  's2', 's3', 's4', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
#     model_name='RUL_attention_FD3_log5_6'
    

# if method=='F1+TaNet':
#     sequence_length=38
#     FD_feature_columns=['s2', 's3','s4','s7', 's8','s9', 's11', 's12', 's13', 's14', 's15', 's17',  's20', 's21']
#     model_name='RUL_attention_FD3_log3_5'
#     # model_name='5 (10)'
    
    
  #  5 6 10 16


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




dataset='cmapssd'
# FD_feature_columns=[  's2', 's3', 's4', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']






# all_feature_columns =['setting1', 'setting2', 'setting3', 's1', 's2', 's3','s4', 's5', 's6', 's7', 's8','s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
# for i_all_feature_columns in range(len(all_feature_columns)):

batch_size=32    
datasets = CMAPSSDataset.CMAPSSDataset(fd_number=FD, batch_size=batch_size, sequence_length=sequence_length,deleted_engine=[1000],feature_columns = FD_feature_columns)#deleted_engine=[5,17,31,41,46,55,69,73,82,95]


train_data = datasets.get_train_data()
train_feature_slice = datasets.get_feature_slice(train_data)
train_label_slice = datasets.get_label_slice(train_data)

# valid_feature_slice = datasets.get_valid_feature_slice(train_data)
# valid_label_slice = datasets.get_valid_label_slice(train_data)



print("train_data.shape: {}".format(train_data.shape))
print("train_feature_slice.shape: {}".format(train_feature_slice.shape))
print("train_label_slice.shape: {}".format(train_label_slice.shape))

# test_data = datasets.get_train_data()    
test_data1 = datasets.get_test_data()
# test_data=test_data.loc[test_data["id"] == 1].head()







def plot_prediction_error(model_name_list,id_num):
    
    x=[]
    y=[]








    
    test_data=test_data1.loc[test_data1['id']==id_num]
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
    
    
    # X_valid=np.transpose( valid_feature_slice,(0,2,1))
    # X_valid=np.reshape(X_valid,(-1,X_valid.shape[1],sequence_length,1))
    # valid_label_slice[valid_label_slice>115]=115
    # Y_valid=valid_label_slice
    
    
    x_test=np.reshape(test_feature_slice,(-1,test_feature_slice.shape[1],1,test_feature_slice.shape[2]))
    y_test_unmaxed=copy.deepcopy(test_label_slice)
    test_label_slice[test_label_slice>115]=115
    y_test=test_label_slice



    model=keras.models.load_model(r"F:\桌面11.17\project\RUL\model\prediction_error\FD{}\{}.h5".format(FD,model_name_list[0]),custom_objects={'root_mean_squared_error': root_mean_squared_error})
    y_test_pred=model.predict(x_test)
    
    print(y_test_unmaxed.min(),y_test_unmaxed.max())
    print(y_test_unmaxed)
    unique_y_test=np.unique(y_test_unmaxed)
    unique_y_test=np.flipud(unique_y_test)
    print(unique_y_test)
    unique_y_test_list=list(unique_y_test)
    y_num_error_np=np.zeros(len(unique_y_test),)
    y_avg_error_np=np.zeros(len(unique_y_test),)
       
    for i in range(len(y_test_unmaxed)): 
        
        if dispaly=='error_abs':        
            predict_error=abs(y_test_pred[i]-y_test[i])#/y_test[i]
            
        if dispaly=='error':            
            predict_error=y_test_pred[i]-y_test[i]
            
        if dispaly=='predict':   
            
            predict_error=y_test_pred[i]
        index=unique_y_test_list.index(y_test_unmaxed[i])
        print(index)

        if i==0:
            
            y_avg_error_np[index]=(y_num_error_np[index]*y_avg_error_np[index]+predict_error)/(y_num_error_np[index]+1)
            
        if i>0:
            
            y_avg_error_np[index]=predict_error+y_avg_error_np[index-1]        
        # y_avg_error_np[index]=(y_num_error_np[index]*y_avg_error_np[index]+predict_error)/(y_num_error_np[index]+1)
        # y_avg_error_np[index]=predict_error
        y_num_error_np[index]=y_num_error_np[index]+1
    
        
    # x.append(unique_y_test) 
    # y.append(y_avg_error_np) 
    
    x_avg=[]
    y_avg=[]
    for i in range(0,len(unique_y_test),avg_len):
        
        x_avg.append(unique_y_test[i])
        y_avg.append(np.mean(np.array(y_avg_error_np[i:i+avg_len])))#(y_num_error_np[i]+y_num_error_np[i+1]+y_num_error_np[i+2]+y_num_error_np[i+3]+y_num_error_np[i+4])/5)
    
        
        
        
        
        


        
    x.append(x_avg) 
    y.append(y_avg) 













    model=keras.models.load_model(r"F:\桌面11.17\project\RUL\model\prediction_error\FD{}\{}.h5".format(FD,model_name_list[1]),custom_objects={'root_mean_squared_error': root_mean_squared_error})
    y_test_pred=model.predict(x_test)
    
    print(y_test_unmaxed.min(),y_test_unmaxed.max())
    print(y_test_unmaxed)
    unique_y_test=np.unique(y_test_unmaxed)
    unique_y_test=np.flipud(unique_y_test)
    print(unique_y_test)
    unique_y_test_list=list(unique_y_test)
    y_num_error_np=np.zeros(len(unique_y_test),)
    y_avg_error_np=np.zeros(len(unique_y_test),)
       
    for i in range(len(y_test_unmaxed)): 
        
        if dispaly=='error_abs':        
            predict_error=abs(y_test_pred[i]-y_test[i])#/y_test[i]
            
        if dispaly=='error':            
            predict_error=y_test_pred[i]-y_test[i]
            
        if dispaly=='predict':   
            
            predict_error=y_test_pred[i]
        index=unique_y_test_list.index(y_test_unmaxed[i])
        print(index)

        if i==0:
            
            y_avg_error_np[index]=(y_num_error_np[index]*y_avg_error_np[index]+predict_error)/(y_num_error_np[index]+1)
            
        if i>0:
            
            y_avg_error_np[index]=predict_error+y_avg_error_np[index-1]        
        # y_avg_error_np[index]=(y_num_error_np[index]*y_avg_error_np[index]+predict_error)/(y_num_error_np[index]+1)
        # y_avg_error_np[index]=predict_error
        y_num_error_np[index]=y_num_error_np[index]+1
    
        
    # x.append(unique_y_test) 
    # y.append(y_avg_error_np) 
    
    x_avg=[]
    y_avg=[]
    for i in range(0,len(unique_y_test),avg_len):
        
        x_avg.append(unique_y_test[i])
        y_avg.append(np.mean(np.array(y_avg_error_np[i:i+avg_len])))#(y_num_error_np[i]+y_num_error_np[i+1]+y_num_error_np[i+2]+y_num_error_np[i+3]+y_num_error_np[i+4])/5)
        
        
        
        


        
    x.append(x_avg) 
    y.append(y_avg) 





    model=keras.models.load_model(r"F:\桌面11.17\project\RUL\model\prediction_error\FD{}\{}.h5".format(FD,model_name_list[2]),custom_objects={'root_mean_squared_error': root_mean_squared_error})
    y_test_pred=model.predict(x_test)
    
    print(y_test_unmaxed.min(),y_test_unmaxed.max())
    print(y_test_unmaxed)
    unique_y_test=np.unique(y_test_unmaxed)
    unique_y_test=np.flipud(unique_y_test)
    print(unique_y_test)
    unique_y_test_list=list(unique_y_test)
    y_num_error_np=np.zeros(len(unique_y_test),)
    y_avg_error_np=np.zeros(len(unique_y_test),)
       
    for i in range(len(y_test_unmaxed)): 
        
        if dispaly=='error_abs':        
            predict_error=abs(y_test_pred[i]-y_test[i])#/y_test[i]
            
        if dispaly=='error':            
            predict_error=y_test_pred[i]-y_test[i]
            
        if dispaly=='predict':   
            
            predict_error=y_test_pred[i]
        index=unique_y_test_list.index(y_test_unmaxed[i])
        print(index)

        if i==0:
            
            y_avg_error_np[index]=(y_num_error_np[index]*y_avg_error_np[index]+predict_error)/(y_num_error_np[index]+1)
            
        if i>0:
            
            y_avg_error_np[index]=predict_error+y_avg_error_np[index-1]        
        # y_avg_error_np[index]=(y_num_error_np[index]*y_avg_error_np[index]+predict_error)/(y_num_error_np[index]+1)
        # y_avg_error_np[index]=predict_error
        y_num_error_np[index]=y_num_error_np[index]+1
    
        
    # x.append(unique_y_test) 
    # y.append(y_avg_error_np) 
    
    x_avg=[]
    y_avg=[]
    for i in range(0,len(unique_y_test),avg_len):
        
        x_avg.append(unique_y_test[i])
        y_avg.append(np.mean(np.array(y_avg_error_np[i:i+avg_len])))#(y_num_error_np[i]+y_num_error_np[i+1]+y_num_error_np[i+2]+y_num_error_np[i+3]+y_num_error_np[i+4])/5)
        
        
        
        
        


        
    x.append(x_avg) 
    y.append(y_avg) 






    
    
    
    
    
    return x,y

if  FD=='1':
    ID=40
    
if  FD=='2':
    ID=121

if  FD=='3':
    ID=46

if  FD=='4':
    ID=7    



for i in range(ID,ID+1):
    id_num=i
    
 #  40,  100 ,  46 , 7

    x,y=plot_prediction_error(['LM+TaNet','LM+A0','LM only'],id_num)
    
    
    xlabel='Actual remaiming life'
    ylabel='Accumulated prediction error'
    
    
    plt.figure() #初始化一张图
    # plt.plot(x[0],y[0],color='y',label='F0+TaNet') #连线图,若要散点图将此句改为：plt.scatter(x,y) #散点图
    plt.plot(x[0],y[0],color='#FF00FF',label='our prognostic method') #连线图,若要散点图将此句改为：plt.scatter(x,y) #散点图
    # plt.plot(x[2],y[2],color='m',label='F1 only') #连线图,若要散点图将此句改为：plt.scatter(x,y) #散点图
    plt.plot(x[1],y[1],color='b',label='SeNet+FCN') #连线图,若要散点图将此句改为：plt.scatter(x,y) #散点图
    # plt.plot(x[4],y[4],color='r',label='LM+A0') #连线图,若要散点图将此句改为：plt.scatter(x,y) #散点图
    # plt.plot(x[5],y[5],color='k',label='LM+A1') #连线图,若要散点图将此句改为：plt.scatter(x,y) #散点图
    # plt.plot(x[6],y[6],color='c',label='LM+A2') #连线图,若要散点图将此句改为：plt.scatter(x,y) #散点图
    plt.plot(x[2],y[2],color='g',label='FCN') #连线图,若要散点图将此句改为：plt.scatter(x,y) #散点图
    plt.grid(alpha=0.5,linestyle='-.') #网格线，更好看
    plt.title('FD'+FD + ' test engine #' +str(ID),fontsize=14) #画总标题 fontsize为字体，下同
    plt.xlabel(xlabel,fontsize=14) #画横坐标
    plt.ylabel(ylabel,fontsize=14) #画纵坐标
    # plt.xlim((0,150))
    
    ax = plt.gca() 
    ax.invert_xaxis()
    # plt.legend(loc="lower right")
    plt.legend()
    plt.savefig(r'F:\桌面11.17\project\RUL\figure\by_kernel\prediction_error{}_{}.eps'.format(FD,id_num),dpi=800,format='eps',bbox_inches = 'tight')
    plt.savefig(r'F:\桌面11.17\project\RUL\figure\by_kernel\prediction_error{}_{}.png'.format(FD,id_num),dpi=800,format='png',bbox_inches = 'tight')

    plt.show() #IDE展示

    



























