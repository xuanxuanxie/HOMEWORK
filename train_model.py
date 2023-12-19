import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio
import numpy as np
import random
import torch.utils.data as data
import os
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from glob import glob
import pandas as pd
from torch.utils.data import Dataset
import scipy.io as io
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# In this section, we will apply an CNN to extract features and implement a classification task.
# Firstly, we should build the model by PyTorch. We provide a baseline model here.
# You can use your own model for better performance
class Doubleconv_33(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Doubleconv_33, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Doubleconv_35(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Doubleconv_35, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=5),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Doubleconv_37(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Doubleconv_37, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=7),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Tripleconv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Tripleconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class MLP(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch_in, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, ch_out),
        )

    def forward(self, input):
        return self.fc(input)


class Mscnn(nn.Module):
    # TODO: Build a better model
    def __init__(self, ch_in, ch_out):
        super(Mscnn, self).__init__()
        self.conv11 = Doubleconv_33(ch_in, 64)
        self.pool11 = nn.MaxPool1d(3, stride=3)
        self.conv12 = Doubleconv_33(64, 128)
        self.pool12 = nn.MaxPool1d(3, stride=3)
        self.conv13 = Tripleconv(128, 256)
        self.pool13 = nn.MaxPool1d(2, stride=2)
        self.conv14 = Tripleconv(256, 512)
        self.pool14 = nn.MaxPool1d(2, stride=2)
        self.conv15 = Tripleconv(512, 512)
        self.pool15 = nn.MaxPool1d(2, stride=2)

        self.out = MLP(512*27, ch_out)  

    def forward(self, x):
        c11 = self.conv11(x)
        p11 = self.pool11(c11)
        c12 = self.conv12(p11)
        p12 = self.pool12(c12)
        c13 = self.conv13(p12)
        p13 = self.pool13(c13)
        c14 = self.conv14(p13)
        p14 = self.pool14(c14)
        c15 = self.conv15(p14)
        p15 = self.pool15(c15)
        merge = p15.view(p15.size()[0], -1) 
        output = self.out(merge)
        output = F.sigmoid(output)
        return output


# Random clipping has been implemented, 
# and you need to add noise and random scaling. 
# Generally, the scaling should be done before the crop.
# In general, do not add scaling and noise enhancement options during testing

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
    

    def load_data(self,file_name,label):
        #读取数据
        mat_file = self.base_file+'/training2017/'+file_name+'.mat'
        data = io.loadmat(mat_file)['val']
        if label=='N':
            one_hot=torch.tensor([0])
        elif label=='O':
            one_hot=torch.tensor([0])
        elif label=='A':
            one_hot=torch.tensor([1])
        elif label=='~':
            one_hot=torch.tensor([0])
        return data,one_hot


    
    def crop_padding(self,data,time):
        #随机crop
        if data.shape[0]<=time:
            data=np.pad(data, (0,time-data.shape[0]), 'constant')
        elif data.shape[0]>time:
            end_index=data.shape[0]-time
            start=np.random.randint(0, end_index)
            data=data[start:start+time]
        return data



    def data_process(self,data):
        # 学习论文以及数据集选择合适和采样率
        # 并完成随机gaussian 噪声和随机时间尺度放缩
        data=data[::3]
        data=data-data.mean()
        data=data/data.std()
        data=self.crop_padding(data,2400)
        data=torch.tensor(data)
        return data


    def __getitem__(self, idx):
        file_name=self.file[idx][1]
        label=self.file[idx][2]
        data,one_hot=self.load_data(file_name,label)
        data=self.data_process(data[0]).unsqueeze(0).float()
        one_hot=one_hot.unsqueeze(0).float()
        return data, one_hot,file_name


# Now, we will build the pipeline for deep learning based training.
# These functions may be useful :)
def save_loss(fold, value):
    path = 'loss' + str(fold) + '.txt'
    file = open(path, mode='a+')
    file.write(str(value)+'\n')  
    
# We will use GPU if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Mscnn(1, 1).to(device)   # ch_in, ch_out

# Build pre-processing transformation 
# Note this pre-processing is in PyTorch
x_transforms = transforms.Compose([
        transforms.ToTensor(),  
])
y_transforms = transforms.ToTensor()


# TODO: fine tune hyper-parameters
batch_size = 64
criterion = torch.nn.MSELoss()
criterion2=torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_ecg_dataset = ECG_dataset('.', is_train=True)
train_dataloader = DataLoader(train_ecg_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_ecg_dataset = ECG_dataset('.', is_train=False)
test_dataloaders = DataLoader(test_ecg_dataset, batch_size=1)
num_epochs = 10


def validation(model,criterion,test_dataloaders,device):
    # TODO: add more metrics for evaluation?
    # Evaluate 
    model.eval()
    predict = np.array([])
    target = np.array([])
    loss=0
    step=0
    with torch.no_grad():
        for x, mask,name in test_dataloaders:
            step += 1
            mask=mask.to(device)
            y = model(x.to(device))
            loss +=criterion(y, mask.squeeze(2)).item()
            y[y >= 0.5] = 1
            y[y < 0.5] = 0
            predict=np.append(predict,torch.squeeze(y).cpu().numpy())
            target=np.append(target,torch.squeeze(mask).cpu().numpy())
    acc = accuracy_score(target, predict)
    print('Accuracy: {}'.format(acc))
    print('Loss:', loss/step)
    model.train()




# Start training !
for epoch in range(1, num_epochs + 1):
        predict = np.array([])
        target = np.array([])
        print('Epoch {}/{}'.format(epoch, num_epochs))
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        process = tqdm(train_dataloader)
        for x, y,name in process:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion2(outputs, labels.squeeze(2))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            process.set_description(
                "epoch: %d, train_loss:%0.8f" % (epoch, epoch_loss / step)
            )
            outputs[outputs >= 0.5] = 1
            outputs[outputs < 0.5] = 0
            predict=np.append(predict,torch.squeeze(outputs).detach().cpu().numpy())
            target=np.append(target,torch.squeeze(labels).detach().cpu().numpy())
        epoch_loss /= step
        acc = accuracy_score(target, predict)
        print('tran_Accuracy: {}'.format(acc))
        save_loss(10, epoch_loss)
        validation(model,criterion2,test_dataloaders,device)
# Save model
torch.save(model.state_dict(), 'weights10_%d.pth' % (epoch))


