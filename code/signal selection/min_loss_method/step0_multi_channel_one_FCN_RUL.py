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



def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true),axis=0))##################  axis=0

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())




    




run_times=3

VALIDATION_SPLIT=0.3

nb_epochs=100             
batch_size=1024   


patience=50
patience_reduce_lr=20


num_filter1=32
num_filter2=64
num_filter3=32

num_filter1=64
num_filter2=128
num_filter3=64




kernel1_size=8
kernel2_size=5
kernel3_size=3

kernel1_size=16
kernel2_size=10
kernel3_size=6




####31,21,38,19
for FD in['3']:
    if FD=='1':
        sequence_length=31
    
    if FD=='2':
        sequence_length=21
    
    if FD=='3':
        sequence_length=38
    if FD=='4':
        sequence_length=19    

    method_name='FD{}_multi_channel_one_FCN_RUL'.format(FD)
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
    

    
    
    all_feature_columns =['setting1', 'setting2', 'setting3', 's1', 's2', 's3','s4', 's5', 's6', 's7', 's8','s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
    for i_all_feature_columns in range(len(all_feature_columns)):
        
        datasets = CMAPSSDataset.CMAPSSDataset(fd_number=FD, batch_size=batch_size, sequence_length=sequence_length,deleted_engine=[1000],feature_columns = all_feature_columns[i_all_feature_columns:i_all_feature_columns+1])#deleted_engine=[5,17,31,41,46,55,69,73,82,95]
        
        
        train_data = datasets.get_train_data()
        train_feature_slice = datasets.get_feature_slice(train_data)
        train_label_slice = datasets.get_label_slice(train_data)
        
        # valid_feature_slice = datasets.get_valid_feature_slice(train_data)
        # valid_label_slice = datasets.get_valid_label_slice(train_data)
        
        
        
        print("train_data.shape: {}".format(train_data.shape))
        print("train_feature_slice.shape: {}".format(train_feature_slice.shape))
        print("train_label_slice.shape: {}".format(train_label_slice.shape))
        
        
        test_data = datasets.get_test_data()
        test_feature_slice, test_label_slice = datasets.get_last_data_slice(test_data)
        
        
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
        
        

        
        X_test=np.reshape(test_feature_slice,(-1,test_feature_slice.shape[1],1,test_feature_slice.shape[2]))
        test_label_slice[test_label_slice>115]=115
        Y_test=test_label_slice
        
        
        print("X_train.shape: {}".format(X_train.shape))
        print("Y_train.shape: {}".format(Y_train.shape))
        
        print("X_test.shape: {}".format(X_test.shape))
        print("Y_test.shape: {}".format(Y_test.shape))
        
        
        import six
        
        import keras.backend as K
        from keras.utils.generic_utils import deserialize_keras_object
        from keras.utils.generic_utils import serialize_keras_object
        from tensorflow.python.ops import math_ops
        from tensorflow.python.util.tf_export import tf_export
        
        
        
        
        
        from tensorflow.python.ops import math_ops
        

        
        
        senet_reduction=1
        dim=11
        

        

        
        

        def FCN_model():
        #    in0 = keras.Input(shape=(sequence_length,train_feature_slice.shape[1]))  # shape: (batch_size, 3, 2048)
        #    in0_shaped= keras.layers.Reshape((train_feature_slice.shape[1],sequence_length,1))(in0)   
        
            in0 = keras.Input(shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3]))  # shape: (batch_size, 3, 2048)
        #    begin_senet=SeBlock()(in0)
            
            x = keras.layers.GlobalAveragePooling2D()(in0)
            x = keras.layers.Dense(int(x.shape[-1]) // 1, use_bias=False,activation=keras.activations.relu)(x)
            kernel = keras.layers.Dense(int(in0.shape[-1]), use_bias=False,activation=keras.activations.hard_sigmoid)(x)
            begin_senet= keras.layers.Multiply()([in0,kernel])    #给通道加权重
         
        
           
        
            conv0 = keras.layers.Conv2D(num_filter1, kernel1_size, strides=1, padding='same')(begin_senet)
            conv0 = keras.layers.BatchNormalization()(conv0)
            conv0 = keras.layers.Activation('relu')(conv0)
            
        #    conv0 = keras.layers.Dropout(dropout)(conv0)
            conv0 = keras.layers.Conv2D(num_filter2, kernel2_size, strides=1, padding='same')(conv0)
            conv0 = keras.layers.BatchNormalization()(conv0)
            conv0 = keras.layers.Activation('relu')(conv0)
            
        #    conv0 = keras.layers.Dropout(dropout)(conv0)
            conv0 = keras.layers.Conv2D(num_filter3, kernel3_size, strides=1, padding='same')(conv0)
            conv0 = keras.layers.BatchNormalization()(conv0)
            conv0 = keras.layers.Activation('relu')(conv0)
            conv0 = keras.layers.GlobalAveragePooling2D()(conv0)
            conv0 = keras.layers.Dense(64, activation='relu')(conv0)
            out = keras.layers.Dense(1, activation='relu')(conv0)
        
            
            
        
        
        
            model = keras.models.Model(inputs=in0, outputs=[out])    
        
            return model
        
        
        # ##############shuaffle the data

        index=np.arange(X_train.shape[0])
        np.random.shuffle(index,)
        
         
        X_train=X_train[index]#X_train是训练集，y_train是训练标签
        Y_train=Y_train[index]
        
        #X_train, Xtest, Y_train, ytest = train_test_split(X_train, Y_train, test_size=0.7, random_state=0)
        
        
        if __name__ == '__main__':
        
            error_record=[]
            index_record=[]
            unbalanced_penalty_score_record=[]
            error_range_record=[]
            index_min_val_loss_record,min_val_loss_record=[],[]
            
            if os.path.exists(r"..\..\..\experiments_result\method_error_txt\{}.txt".format(method_name)):os.remove(r"..\..\..\experiments_result\method_error_txt\{}.txt".format(method_name))
        
         
      
        
        
               
                        
        #######             single  output                
        
            for i in range(run_times):
                print('xxx')
            
                model=FCN_model()
                
                optimizer = keras.optimizers.Adam()
                model.compile(loss='mse',#loss=root_mean_squared_error,
                              optimizer=optimizer,
                              metrics=[root_mean_squared_error])
                 
    
                

    
                hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epochs, verbose=1)   
        #        hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
        #                  verbose=1, validation_data=(X_test, Y_test), callbacks = [reduce_lr,earlystopping,modelcheckpoint])   
                log = pd.DataFrame(hist.history)
                log.to_excel(r"..\..\..\experiments_result\log\feature_select_valid0\{}_{}_dataset_{}_log{}_time{}.xlsx".format(method_name,all_feature_columns[i_all_feature_columns],dataset,i,datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
                
                print(hist.history.keys())
                epochs=range(len(hist.history['loss']))
                plt.figure()
                plt.plot(epochs,hist.history['loss'],'b',label='Training loss')
    
                plt.title('Traing and Validation loss')
                plt.legend()
                plt.show()




