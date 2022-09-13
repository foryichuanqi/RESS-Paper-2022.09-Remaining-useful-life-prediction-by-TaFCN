import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# sequence_length is equal to timesteps
# the file path of dataset

columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3','s4', 's5', 's6', 's7', 's8',
         's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

feature_columns = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3','s4', 's5', 's6', 's7', 's8',
         's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

# delete_list=[24,22,21,20,18,16,12,8,7,3,2]

# for i in range(len(delete_list)):
#     del feature_columns[delete_list[i]]
    
print(feature_columns)
    #['setting1', 'setting2', 's2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's15', 's17', 's21']

class CMAPSSDataset():
    def __init__(self, fd_number, batch_size, sequence_length,deleted_engine,feature_columns):
        super(CMAPSSDataset).__init__()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.train_data = None
        self.test_data = None
        self.deleted_engine=deleted_engine
        self.feature_columns=feature_columns
        
        # read train_FD00x
        data = pd.read_csv("..\C-MAPSS-Data\\train_FD00" + fd_number + ".txt", delimiter="\s+", header=None)
        data.columns = columns
        self.engine_size = max(data['id'])


        
        # Calculate rul
        rul = pd.DataFrame(data.groupby('id')['cycle'].max()).reset_index()
        rul.columns = ['id', 'max']
        print(rul)
        data = data.merge(rul, on=['id'], how='left')
        data['RUL'] = data['max'] - data['cycle']
        data.drop(['max'], axis=1, inplace=True)
        
        # Normalize columns other than 'id', 'cycle', 'RUL'
        self.std = StandardScaler()
        data['cycle_norm'] = data['cycle']
        cols_normalize = data.columns.difference(['id', 'cycle', 'RUL'])
        norm_data = pd.DataFrame(self.std.fit_transform(data[cols_normalize]), columns=cols_normalize, index=data.index)
        join_data = data[data.columns.difference(cols_normalize)].join(norm_data)
        self.train_data = join_data.reindex(columns=data.columns)
        
        # Read the test dataset by the RUL_FD00x.txt file.
        test_data = pd.read_csv("..\C-MAPSS-Data\\test_FD00" + fd_number + ".txt", delimiter="\s+", header=None)
        test_data.columns = columns
        truth_data = pd.read_csv("..\C-MAPSS-Data\\RUL_FD00" + fd_number + ".txt", delimiter="\s+", header=None)
        truth_data.columns = ['truth']
        truth_data['id'] = truth_data.index + 1
        
        test_rul = pd.DataFrame(test_data.groupby('id')['cycle'].max()).reset_index()
        test_rul.columns = ['id', 'elapsed']
        test_rul = test_rul.merge(truth_data, on=['id'], how='left')
        test_rul['max'] = test_rul['elapsed'] + test_rul['truth']
        
        test_data = test_data.merge(test_rul, on=['id'], how='left')
        test_data['RUL'] = test_data['max'] - test_data['cycle']
        test_data.drop(['max'], axis=1, inplace=True)
        
        test_data['cycle_norm'] = test_data['cycle']
        norm_test_data = pd.DataFrame(self.std.transform(test_data[cols_normalize]), columns=cols_normalize, index=test_data.index)
        join_test_data = test_data[test_data.columns.difference(cols_normalize)].join(norm_test_data)
        self.test_data = join_test_data.reindex(columns=test_data.columns)
     
    def get_train_data(self):
        return self.train_data
    
    def get_test_data(self):
        return self.test_data
        

    
    def get_feature_slice(self, data):
        feature_list=[]
        for i in range(1, self.engine_size + 1):
            if i in self.deleted_engine  :
                continue
            selected_feature_data=data[data['id'] == i][self.feature_columns].values
            # print(selected_feature_data.shape)
            for j in range(0,selected_feature_data.shape[0]-self.sequence_length+1):
                feature_list.append(selected_feature_data[j:j+self.sequence_length,:])
        feature_array = np.array(feature_list).astype(np.float32)

        return feature_array ############to get integral multiple of batch_size, shape(samples, time steps, features)
    



    def get_label_slice(self, data):
        
        label_list=[]
        for i in range(1, self.engine_size + 1):
            if i in self.deleted_engine  :
                continue
            selected_label_data=data[data['id'] == i]['RUL'].values
            # print(selected_label_data.shape)
            for j in range(0,selected_label_data.shape[0]-self.sequence_length+1):
                label_list.append(selected_label_data[j+self.sequence_length-1])
        lable_array = np.array(label_list).astype(np.float32).reshape(-1,1)

        return lable_array ############




    def get_last_data_slice(self, data):
        

        
        feature_list=[]
        for i in range(1, self.engine_size + 1):
            if i in self.deleted_engine  :
                continue
            selected_feature_data=data[data['id'] == i][self.feature_columns].values
            # print(selected_feature_data.shape)
            for j in range(0,selected_feature_data.shape[0]-self.sequence_length+1):
                if j != selected_feature_data.shape[0]-self.sequence_length:
                    continue
                feature_list.append(selected_feature_data[j:j+self.sequence_length,:])
        test_feature_array = np.array(feature_list).astype(np.float32)
        


        label_list=[]
        for i in range(1, self.engine_size + 1):
            if i in self.deleted_engine  :
                continue
            selected_label_data=data[data['id'] == i]['RUL'].values
            # print(selected_label_data.shape)
            for j in range(0,selected_label_data.shape[0]-self.sequence_length+1):
                if j != selected_label_data.shape[0]-self.sequence_length:
                    continue
                label_list.append(selected_label_data[j+self.sequence_length-1])
        test_label_array = np.array(label_list).astype(np.float32).reshape(-1,1)
        
        return test_feature_array, test_label_array############
    
    




if __name__ == "__main__":

    FD='1'
    num_test=100
    
          
    batch_size=1024   
    
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


    datasets = CMAPSSDataset(fd_number=FD, batch_size=batch_size, sequence_length=sequence_length,deleted_engine=[1000],feature_columns = FD_feature_columns)#deleted_engine=[5,17,31,41,46,55,69,73,82,95]
    
    
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



