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


#11111111111111111111111111111111111111111111111111
#NOTE  NOTE  NOTE  NOTE  NOTE  NOTE  NOTE  NOTE
#11111111111111111111111111111111111111111111111111
#NOTE  NOTE  NOTE  NOTE  NOTE  NOTE  NOTE  NOTE
#11111111111111111111111111111111111111111111111111
#NOTE  NOTE  NOTE  NOTE  NOTE  NOTE  NOTE  NOTE
#11111111111111111111111111111111111111111111111111
#NOTE  NOTE  NOTE  NOTE  NOTE  NOTE  NOTE  NOTE
#11111111111111111111111111111111111111111111111111
#NOTE  NOTE  NOTE  NOTE  NOTE  NOTE  NOTE  NOTE

# note:to reduce the size of the compressed package, we only provide the model of LM+TaNet. 

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


num_test=100


FD='LM+TaNet'




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
if FD=='LM+TaNet':
    sequence_length=38
    FD_feature_columns=[  's2', 's3', 's4', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    model_name='RUL_attention_FD3_log5_6'
    

if FD=='F1+TaNet':
    sequence_length=38
    FD_feature_columns=['s2', 's3','s4','s7', 's8','s9', 's11', 's12', 's13', 's14', 's15', 's17',  's20', 's21']
    model_name='RUL_attention_FD3_log3_5'
    # model_name='5 (10)'
    
    
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
datasets = CMAPSSDataset.CMAPSSDataset(fd_number='3', batch_size=batch_size, sequence_length=sequence_length,deleted_engine=[1000],feature_columns = FD_feature_columns)#deleted_engine=[5,17,31,41,46,55,69,73,82,95]


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




x_test=np.reshape(test_feature_slice,(-1,test_feature_slice.shape[1],1,test_feature_slice.shape[2]))
test_label_slice[test_label_slice>115]=115
y_test=test_label_slice



print("X_train.shape: {}".format(X_train.shape))
print("Y_train.shape: {}".format(Y_train.shape))

print("X_test.shape: {}".format(x_test.shape))
print("Y_test.shape: {}".format(y_test.shape))
        

model=keras.models.load_model(r"..\..\model\interpretability_TaNet\{}.h5".format(model_name),custom_objects={'root_mean_squared_error': root_mean_squared_error})

############## Get CAM ################
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages




get_kernel = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], [model.layers[-14].output])
kernel = get_kernel([x_test[:100], 1])[0]



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

##############read  table_data from xlsx
from scipy import stats
import numpy as np
import pandas as pd





plt.boxplot(x = kernel, # 指定绘图数据
            patch_artist  = True, # 要求用自定义颜色填充盒形图，默认白色填充
            showmeans = True, # 以点的形式显示均值
            widths = 0.5,
            boxprops = {'color':'black','facecolor':'#9999ff'}, # 设置箱体属性，填充色和边框色
            flierprops = {'marker':'o','markerfacecolor':'red','color':'black'}, # 设置异常值属性，点的形状、填充色和边框色
            meanprops = {'marker':'D','markerfacecolor':'indianred'}, # 设置均值点的属性，点的形状、填充色
            medianprops = {'linestyle':'--','color':'orange'}) # 设置中位数线的属性，线的类型和颜色




if FD=='LBFS+TaNet':

    plt.axhline(y=0.6, color='y', linestyle='-')
    
    plt.axhline(y=0.4, color='y', linestyle='-')
    
    plt.axhline(y=0.3, color='y', linestyle='-')
    

if FD=='F1+TaNet':



    plt.axhline(y=0.4, color='y', linestyle='-')






plt.xticks(range(1,len(FD_feature_columns)+1),FD_feature_columns,color='blue',rotation=90)

plt.grid(ls='--')

#设置图例并且设置图例的字体及大小
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 14,
}

  
#设置横纵坐标的名称以及对应字体格式
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 30,
}

plt.xlabel('sensor',font1) #X轴标签
plt.ylabel("attention value",font1) #Y轴标签
plt.title(FD)
# foo_fig = plt.gcf()
# foo_fig.savefig(r'F:\桌面11.17\project\fluid_based_time_series_calssification\figure\box_compare_figure.eps',dpi=800,format='eps',bbox_inches = 'tight')
# plt.savefig(r'F:\桌面11.17\project\fluid_based_time_series_calssification\figure\box_compare_figure.eps', dpi=150)
plt.savefig(r'..\..\figure\by_kernel\interpretability_TaNet_analysis_{}_{}.eps'.format(FD,model_name),dpi=800,format='eps',bbox_inches = 'tight')
plt.savefig(r'..\..\figure\by_kernel\interpretability_TaNet_analysis_{}_{}.png'.format(FD,model_name),dpi=800,format='png',bbox_inches = 'tight')

plt.show()


