import pandas as pd
from scipy import io
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from scipy import signal

class ECG_dataset(Dataset):

    def __init__(self,base_file,cv=0, is_train=True):

        self.is_train = is_train
        self.file_list=[]
        self.base_file=base_file
        
        for i in range(5):
            data=pd.read_csv(base_file+'/cv/cv'+str(i)+'.csv')
            self.file_list.append(data.to_numpy())
        self.file=None
        if is_train:
            del self.file_list[cv]
            self.file=self.file_list[0]
            for i in range(1,4):
                self.file=np.append(self.file,self.file_list[i],axis=0)
        else:
            self.file=self.file_list[cv]

        
    def __len__(self):
        return self.file.shape[0]
    
    def load_data_for_svm(self, file_name, label):
    # Load data
        mat_file = self.base_file + '/training2017/' + file_name + '.mat'
        data = io.loadmat(mat_file)['val']

    # Resample data from 1000Hz to 250Hz
        data = signal.resample(data, 750)

    # Flatten the data
        data = data.flatten()

    # Convert label to integer
        if label == 'N' or label == 'O':
            label = 0
        elif label == 'A':
            label = 1
        else:  # label == '~'
            label = 2

        return data, label
    

    def crop_padding(self, data, time):
        if data.shape[0] <= time:
            data = np.pad(data, (0, time - data.shape[0]), 'constant')
        elif data.shape[0] > time:
            end_index = data.shape[0] - time
            start = np.random.randint(0, end_index)
            data = data[start:start + time]
        return data

    def __getitem__(self, idx):
        file_name=self.file[idx][1]
        label=self.file[idx][2]
        data,one_hot=self.load_data(file_name,label)
        data=self.data_process(data[0]).unsqueeze(0).float()
        one_hot=one_hot.unsqueeze(0).float()
        return data, one_hot,file_name
    

X = []
y = []

for file_name, label in zip(file_names, labels):
    data, label = load_data_for_svm(file_name, label)
    X.append(data)
    y.append(label)

X = np.array(X)
y = np.array(y)

