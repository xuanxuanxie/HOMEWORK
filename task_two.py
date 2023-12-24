import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import scipy.io as io
import torch
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader




class ECG_SVM_dataset(Dataset):

    def __init__(self, base_file=None, cv=0, is_train=True):
        self.is_train = is_train
        self.file_list = []
        self.base_file = base_file
        
        for i in range(5):
            data = pd.read_csv(base_file+'/cv/cv'+str(i)+'.csv')
            self.file_list.append(data.to_numpy())
        self.file = None
        if is_train:
            del self.file_list[cv]
            self.file = self.file_list[0]
            for i in range(1, 4):
                self.file = np.append(self.file, self.file_list[i], axis=0)
        else:
            self.file = self.file_list[cv]

    def __len__(self):
        return self.file.shape[0]

    def load_data(self, file_name, label):
        mat_file = self.base_file+'/training2017/'+file_name+'.mat'
        data = io.loadmat(mat_file)['val']
        return data, label

    def __getitem__(self, idx):
        file_name = self.file[idx][1]
        label = self.file[idx][2]
        data, label = self.load_data(file_name, label)
        return data, label, file_name


data=ECG_SVM_dataset('.')
d=data[1]
print(d)
d_0 = d[0]
x = np.linspace(0,len(d[0]), len(d[0])) # x轴坐标值
plt.plot(x, d[0],c = 'r') # 参数c为color简写，表示颜色,r为red即红色
plt.show() # 显示图像
print("^^")