# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 12:24:37 2021

@author: flc
"""
##FD
from sklearn import preprocessing
import numpy as np
 




########## paratactic hitogram
def negative_to_zero(original_list):
    for i in range(len(original_list)):
        if original_list[i]<0:
            original_list[i]=0
    return original_list

import numpy as np
import matplotlib.pyplot as plt


# weighted_value=[3.036197968089715e-10, 0.9963760952393138, 0.9963753136093428, 0.9940911689858869, 0.7808021987771123, 0.8180642356305036, 0.9930201676519741, 0.9893208197740085, 0.9774663259165547, 3.036197968089715e-10, 0.9948905088031987, 0.9927159275488282, 0.9923477959204415, 0.9800775393319126, 0.9951745010866035, 0.7808021987771123, 0.9964176507907656, 3.036197968089715e-10, 3.036197968089715e-10, 0.9950692112458714, 0.9936094027304543]
weighted_value=[ -1.3525545291981995, 0.8580708041129999, 0.8432959040345853, 0.8185273692842254, -0.08518421240324271, 0.014344690169709321, 0.8080746736887614, 0.7464706886790279, 0.634078690879124, -1.3525545291981995, 0.8293482132538018, 0.7959397025237085, 0.7769987758394079, 0.6557361651410154, 0.8319090714471812, -0.08518421240324271, 0.8503126509071743, -1.3525545291981995, -1.3525545291981995, 0.8332705563081861, 0.8112508345938821]
weighted_value=negative_to_zero(weighted_value)



waters= ('s1', 's2', 's3','s4', 's5', 's6', 's7', 's8',
        's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21')


    
 
plt.bar(waters, weighted_value)

plt.xticks(range(0,len(waters)),waters,color='blue',rotation=60)
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

plt.xlabel('feature',font1) #X轴标签
plt.ylabel("mp",font1) #Y轴标签
plt.title('FD1',font1)
plt.savefig(r'F:\桌面11.17\project\RUL\figure\by_kernel\feature_select_FD1.eps',dpi=800,format='eps',bbox_inches = 'tight')
plt.savefig(r'F:\桌面11.17\project\RUL\figure\by_kernel\feature_select_FD1.png',dpi=800,format='png',bbox_inches = 'tight')

plt.show()




import numpy as np
import matplotlib.pyplot as plt


weighted_value=[ 0.6349856534065781, 0.6345172254497131, 0.7280255036646421, 0.7704532243067866, 0.6532500146918113, 0.635776697712685, 0.6322822645235303, 0.6114007857498442, 0.6549378349486613, 0.6139399959958015, 0.7652670869094798, 0.6338647669024404, 0.06011263052365457, 0.6250947719475386, 0.7146342024839003, 0.4415509624016986, 0.7288263251986141, 0.6421262526277283, 0.047940830293252766, 0.6449865864287931, 0.660358380479112]
weighted_value=negative_to_zero(weighted_value)
waters= ('s1', 's2', 's3','s4', 's5', 's6', 's7', 's8',
        's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21')


    
 
plt.bar(waters, weighted_value)

plt.xticks(range(0,len(waters)),waters,color='blue',rotation=60)
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

plt.xlabel('feature',font1) #X轴标签
plt.ylabel("mp",font1) #Y轴标签
plt.title('FD2',font1)
plt.savefig(r'F:\桌面11.17\project\RUL\figure\by_kernel\feature_select_FD2.eps',dpi=800,format='eps',bbox_inches = 'tight')
plt.savefig(r'F:\桌面11.17\project\RUL\figure\by_kernel\feature_select_FD2.png',dpi=800,format='png',bbox_inches = 'tight')

plt.show()



import numpy as np
import matplotlib.pyplot as plt


weighted_value=[-1.5574784656475686, 0.8316110515289997, 0.841343175823377, 0.8285534873681247, -0.09808395402285239, 0.1407054700381496, 0.7127104062989118, 0.6109340608867988, 0.7037998406043584, 0.07869900923286073, 0.770199743253822, 0.6844810775917795, 0.6976549117990988, 0.6692217970981309, 0.7829009830050208, -0.09808395402285239, 0.8066354612199966, -1.5574784656475686, -1.5574784656475686, 0.7671878545340377, 0.7679127656271648]
weighted_value=negative_to_zero(weighted_value)
waters= ('s1', 's2', 's3','s4', 's5', 's6', 's7', 's8',
        's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21')


    
 
plt.bar(waters, weighted_value)

plt.xticks(range(0,len(waters)),waters,color='blue',rotation=60)
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

plt.xlabel('feature',font1) #X轴标签
plt.ylabel("mp",font1) #Y轴标签
plt.title('FD3',font1)
plt.savefig(r'F:\桌面11.17\project\RUL\figure\by_kernel\feature_select_FD3.eps',dpi=800,format='eps',bbox_inches = 'tight')
plt.savefig(r'F:\桌面11.17\project\RUL\figure\by_kernel\feature_select_FD3.png',dpi=800,format='png',bbox_inches = 'tight')

plt.show()




import numpy as np
import matplotlib.pyplot as plt

weighted_value=[ 0.5089387560755972, 0.5121675290589925, 0.7240903718579894, 0.7440636647722043, 0.5126518120078876, 0.506333457562931, 0.507563206383561, 0.4776631277837427, 0.6279954541821787, 0.5500065837118352, 0.7234265002200237, 0.5021685035072925, 0.24183469235576593, 0.6264476558593762, 0.6248106457731102, 0.2984575273162594, 0.7227375654379092, 0.47479686150199546, -0.016127577443375733, 0.5089173115135502, 0.5070780838586685]
weighted_value=negative_to_zero(weighted_value)

waters= ('s1', 's2', 's3','s4', 's5', 's6', 's7', 's8',
        's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21')


    
 
plt.bar(waters, weighted_value)

plt.xticks(range(0,len(waters)),waters,color='blue',rotation=60)
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

plt.xlabel('feature',font1) #X轴标签
plt.ylabel("mp",font1) #Y轴标签
plt.title('FD4',font1)
plt.savefig(r'F:\桌面11.17\project\RUL\figure\by_kernel\feature_select_FD4.eps',dpi=800,format='eps',bbox_inches = 'tight')
plt.savefig(r'F:\桌面11.17\project\RUL\figure\by_kernel\feature_select_FD4.png',dpi=800,format='png',bbox_inches = 'tight')

plt.show()




