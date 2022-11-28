# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 22:23:26 2021

@author: Administrator
"""


#import tensorflow as tf

import os

 
print(os.path.abspath(os.path.join(os.getcwd(), "../..")))
last_last_path=os.path.abspath(os.path.join(os.getcwd(), "../.."))

print(os.path.abspath(os.path.join(os.getcwd(), "..")))
last_path=os.path.abspath(os.path.join(os.getcwd(), ".."))

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




    


num_test=100

run_times=10



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
for FD in['1','2','4','3']: ######['1','2','3','4']


    
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





    method_name='grid_FD{}_A0_num_test{}'.format(FD,num_test)
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
     
    
       
    

    
        conv0 = keras.layers.Conv2D(num_filter1, kernel1_size, strides=1, padding='same',name='layer_9')(begin_senet)
        conv0 = keras.layers.BatchNormalization()(conv0)
        conv0 = keras.layers.Activation('relu',name='layer_8')(conv0)
        
    #    conv0 = keras.layers.Dropout(dropout)(conv0)
        conv0 = keras.layers.Conv2D(num_filter2, kernel2_size, strides=1, padding='same',name='layer_7')(conv0)
        conv0 = keras.layers.BatchNormalization()(conv0)
        conv0 = keras.layers.Activation('relu',name='layer_6')(conv0)
        
    #    conv0 = keras.layers.Dropout(dropout)(conv0)
        conv0 = keras.layers.Conv2D(num_filter3, kernel3_size, strides=1, padding='same',name='layer_5')(conv0)
        conv0 = keras.layers.BatchNormalization()(conv0)
        conv0 = keras.layers.Activation('relu',name='layer_4')(conv0)
        conv0 = keras.layers.GlobalAveragePooling2D(name='layer_3')(conv0)
        conv0 = keras.layers.Dense(64, activation='relu',name='layer_2')(conv0)
        out = keras.layers.Dense(1, activation='relu',name='layer_1')(conv0)            
        
    
    
    
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
        error_range_left_record=[]
        error_range_right_record=[]
        index_min_val_loss_record,min_val_loss_record=[],[]
        
        if os.path.exists(last_path+r"\experiments_result\method_error_txt\{}.txt".format(method_name)):os.remove(last_path+r"\experiments_result\method_error_txt\{}.txt".format(method_name))
   
     
    
   
    
    
    
    
           
                    
    #######             single  output                
    
        for i in range(run_times):

        
            model=FCN_model()
            
            optimizer = keras.optimizers.Adam()
            model.compile(loss='mse',#loss=root_mean_squared_error,
                          optimizer=optimizer,
                          metrics=[root_mean_squared_error])
             
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                              patience=patience_reduce_lr, min_lr=0.0001) 
            
  
            model_name='{}_dataset_{}_log{}_time{}'.format(method_name,dataset,i,datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
            earlystopping=keras.callbacks.EarlyStopping(monitor='loss',patience=patience,verbose=1)
            modelcheckpoint=keras.callbacks.ModelCheckpoint(monitor='loss',filepath=last_path+r"\model\FCN_RUL_1out_train_valid_test\{}.h5".format(model_name),save_best_only=True,verbose=1)
            hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
                      verbose=1, validation_data=(X_test, Y_test), callbacks = [reduce_lr,earlystopping,modelcheckpoint])   
    #        hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
    #                  verbose=1, validation_data=(X_test, Y_test), callbacks = [reduce_lr,earlystopping,modelcheckpoint])   
            log = pd.DataFrame(hist.history)
            log.to_excel(last_path+r"\experiments_result\log\{}_dataset_{}_log{}_time{}.xlsx".format(method_name,dataset,i,datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
            
            print(hist.history.keys())
            epochs=range(len(hist.history['loss']))
            plt.figure()
            plt.plot(epochs,hist.history['loss'],'b',label='Training loss')
            plt.plot(epochs,hist.history['val_loss'],'r',label='Validation val_loss')
            plt.title('Traing and Validation loss')
            plt.legend()
            plt.show()
    
            
            model=keras.models.load_model(last_path+r"\model\FCN_RUL_1out_train_valid_test\{}.h5".format(model_name),custom_objects={'root_mean_squared_error': root_mean_squared_error})
            for layer in model.layers:
                layer.trainable=False        
    #        score = model.evaluate(X_test, Y_test)  ############forbid evaluate!!!!!!!!!!!!!!!!!!
    #        print('score[1]:{}'.format(score[1]))    ############forbid evaluate!!!!!!!!!!!!!!!!!!    
            
            Y_pred=model.predict(X_test)
    #        rmse=root_mean_squared_error(Y_test,Y_pred)
    #        with tf.Session() as sess:
    #            print(rmse.eval())
            rmse_value=rmse(Y_test,Y_pred)
            print('rmse:{}'.format(rmse_value))
    
                
            unbalanced_penalty_score=unbalanced_penalty_score_1out(Y_test,Y_pred)
            error_range=error_range_1out(Y_test,Y_pred)             
            
            
            # log = pd.DataFrame(hist.history)
            # log.to_excel(r"F:\桌面11.17\project\RUL\experiments_result\log\{}_dataset_{}_log{}_time{}.xlsx".format(method_name,dataset,i,datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
            
            index_min_val_loss,min_val_loss=log['loss'].idxmin(axis=1), log.loc[log['loss'].idxmin]['val_loss']
            print(index_min_val_loss)
            index_min_val_loss_record.append(index_min_val_loss)
            min_val_loss_record.append(min_val_loss)
            error_record.append(rmse_value)
            unbalanced_penalty_score_record.append(unbalanced_penalty_score)
            error_range_left_record.append(error_range[0])
            error_range_right_record.append(error_range[1])
    

            
            file = open(last_path+r"\experiments_result\method_error_txt\{}.txt".format(method_name), 'a')
            file.write( '       ('+str(i)+ ')   '+'index_min_loss:'+str(index_min_val_loss)+'        min_val_loss:'+str(min_val_loss)+'     RMSE:'+'    '+str('%.6f'%(rmse_value))+'     '+'UPE:'+'    '+str('%.6f'%(unbalanced_penalty_score))+'    '+'ER:'+'('+str('%.6f'%(error_range[0]))+','+str('%.6f'%(error_range[1]))+')')
         
    #        file.write( 'error:'+'   '+str('%.5f'%(1-log.loc[log['loss'].idxmin]['val_acc']))+'        '+'corresponding_min_loss:'+'   '+str('%.5f'%log.loc[log['loss'].idxmin]['loss']) +'        '+str(flist.index(each))+'        ' +each +'\n')
    #        file.write( 'error:'+'   '+str('%.5f'%(1-log.loc[log['loss'].idxmin]['val_acc']))+'        '+'corresponding_min_loss:'+'   '+str('%.5f'%log.loc[log['loss'].idxmin]['loss']) +'        '+str(flist.index(each))+'        ' +each +'\n')
        
            file.close()
    
            # print('!!!!!!!!!!!!!!!!  {} {} {}::runtime:{}____min_error:{}'.format(method_name,num,each,i,'%.5f'%min(error_record)))
    
        r0=[]
        r1=[]
        r2=[]
        r3=[]                        
        for i in range(len(error_record)):
            
            if error_record[i]<40:
                r0.append(unbalanced_penalty_score_record[i])
                r1.append(error_range_left_record[i])
                r2.append(error_range_right_record[i])
                r3.append(error_record[i])
        unbalanced_penalty_score_record=r0
        error_range_left_record=r1
        error_range_right_record=r2
        error_record=r3
        
        file = open(last_path+r"\experiments_result\method_error_txt\{}.txt".format(method_name), 'a')
        file.write('    mean_score:'+'     ('+str(np.mean(error_record))+')     '+'       '+'     mean_RMSE:   '+ str('%.6f'%(np.mean(error_record)))+'     '+'UPE:    '+ str('%.6f'%(np.mean(unbalanced_penalty_score_record)))+ '    ('+str('%.6f'%(np.mean(error_range_left_record)))+','+str('%.6f'%(np.mean(error_range_right_record)))+')'+'        '  +'\n')
        file.close()
        error_record=[]
    
        unbalanced_penalty_score_record=[]
        error_range_left_record=[]
        error_range_right_record=[]





