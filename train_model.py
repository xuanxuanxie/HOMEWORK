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
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, matthews_corrcoef, balanced_accuracy_score
from scipy.signal import resample
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


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

        input = input.squeeze(2)  # Remove the extra dimension
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
        self.fc = nn.Linear(ch_out * 27, 2)
        self.bn = nn.BatchNorm1d(ch_out)  # Add batch normalization layer

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x.view(x.size(0), -1))  # Apply batch normalization
        return self.fc(x)

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

class Doubleconv_Stream2(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size):
        super(Doubleconv_Stream2, self).__init__()
        padding = kernel_size // 2  # 保持输出尺寸不变
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=kernel_size, padding=padding, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=kernel_size, padding=padding, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Mscnn(nn.Module):
    # TODO: Build a better model
    def __init__(self, ch_in, ch_out, kernel_size_stream2=5):# stream2的卷积核大小为5，修改这个参数即可
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

        # self.out = MLP(512*27, ch_out)  

        self.stream2_conv1 = Doubleconv_Stream2(ch_in, 64, kernel_size_stream2)
        self.stream2_pool1 = nn.MaxPool1d(3, stride=3)
        self.stream2_conv2 = Doubleconv_Stream2(64, 128, kernel_size_stream2  )
        self.stream2_pool2 = nn.MaxPool1d(3, stride=3)
        self.stream2_conv3 = Doubleconv_Stream2(128, 256, kernel_size_stream2  )
        self.stream2_pool3 = nn.MaxPool1d(2, stride=2)
        self.stream2_conv4 = Doubleconv_Stream2(256, 512, kernel_size_stream2)
        self.stream2_pool4 = nn.MaxPool1d(2, stride=2)
        self.stream2_conv5 = Doubleconv_Stream2(512, 512, kernel_size_stream2 )
        self.stream2_pool5 = nn.MaxPool1d(2, stride=2)
        
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
        stream2_output = self.stream2_conv1(x)
        stream2_output = self.stream2_pool1(stream2_output)
        stream2_output = self.stream2_conv2(stream2_output)
        stream2_output = self.stream2_pool2(stream2_output)
        stream2_output = self.stream2_conv3(stream2_output)
        stream2_output = self.stream2_pool3(stream2_output)
        stream2_output = self.stream2_conv4(stream2_output)
        stream2_output = self.stream2_pool4(stream2_output)
        stream2_output = self.stream2_conv5(stream2_output)
        stream2_output = self.stream2_pool5(stream2_output)
        print("Stream1 output shape:", p15.shape)
        print("Stream2 output shape:", stream2_output.shape)
        # merged_output = p15 + stream2_output
       
       
       
        # 在合并操作之前添加额外的池化步骤
        
        if stream2_output.size(2) > p15.size(2):
            stream2_output = stream2_output[:, :, :p15.size(2)]
       

        # 合并操作
        merged_output = torch.cat((p15, stream2_output), dim=1)
        size1 = merged_output.size(1) // 2
        merged_output = merged_output[:,:size1]
        print("Merged output shape:", merged_output.shape)
        final_output = self.out(merged_output.view(merged_output.size(0), -1))
        final_output = F.sigmoid(final_output)   
        return final_output
        # merge = p15.view(p15.size()[0], -1) 
        # output = self.out(merge)
        # output = F.sigmoid(output)
        # return output


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
        # 假设原始采样率是1000Hz，目标采样率是250Hz
        if label=='N':
            one_hot=torch.tensor([0])
        elif label=='O':
            one_hot=torch.tensor([0])
        elif label=='A':
            one_hot=torch.tensor([1])
        elif label=='~':
            one_hot=torch.tensor([0])
        return data,one_hot

    
    # def crop_padding(self,data,time):
    #     #随机crop
    #     if data.shape[0]<=time:
    #         data=np.pad(data, (0,time-data.shape[0]), 'constant')
    #     elif data.shape[0]>time:
    #         end_index=data.shape[0]-time
    #         start=np.random.randint(0, end_index)
    #         data=data[start:start+time]
    #     return data
    def crop_padding(self, data, time):
        # 随机缩放
        # scale_factor = random.uniform(0.8, 1.2)
        # data = torch.nn.functional.interpolate(data.unsqueeze(0).unsqueeze(0), scale_factor=scale_factor, mode='linear').squeeze()
       

        # 随机crop
        if data.shape[0] <= time:
            data = np.pad(data, (0, time - data.shape[0]), 'constant')
        elif data.shape[0] > time:
            end_index = data.shape[0] - time
            start = np.random.randint(0, end_index)
            data = data[start:start + time]
        return data



    def data_process(self,data):
        # 学习论文以及数据集选择合适和采样率 #  ------------------------已完成--------------------------
        # 并完成随机gaussian 噪声和随机时间尺度放缩  #  ---------------------------已完成高斯噪声和尺度----------
        data=data[::3]#修改第二个数字修改采样率训练
        data=data-data.mean()
        data=data/data.std()
        if self.is_train:# 仅在训练时添加噪声和进行缩放，验证和测试数据集上，不会应用这些增强操作。
            noise = torch.randn(data.shape) * random.uniform(0.1, 0.5)
            noise = noise.numpy()
            data = data + noise

            data = torch.from_numpy(data)  # Convert to PyTorch tensor
            scale_factor = random.uniform(0.8, 1.2)
            data = torch.nn.functional.interpolate(data.unsqueeze(0).unsqueeze(0), scale_factor=scale_factor, mode='linear').squeeze()

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
batch_size = 64 #  -----------------------原batch_size=64，修改超参数之批次大小--------------------------
# criterion = torch.nn.MSELoss()# 损失函数为均方误差，此为原函数
criterion = torch.nn.CrossEntropyLoss()  # 如果是多分类问题
criterion2=torch.nn.BCELoss()# 二元交叉熵损失（BCELoss）是用于二分类问题的损失函数，而交叉熵损失（CrossEntropyLoss）是用于多分类问题的损失函数
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #Adam优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # 使用SGD优化器


train_ecg_dataset = ECG_dataset('.', is_train=True)
train_dataloader = DataLoader(train_ecg_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_ecg_dataset = ECG_dataset('.', is_train=False)
test_dataloaders = DataLoader(test_ecg_dataset, batch_size=1)
num_epochs = 1#  -----------------------原num_epochs=20，修改超参数之epoch轮数--------------------------

f1_scores = []
auc_scores = []
losses = []
precisions = []
recalls = []
mccs = []
balanced_accuracies = []


def validation(model,criterion,test_dataloaders,device):
    # TODO: add more metrics for evaluation?
    # Evaluate   #-----------------------加了F1分数和AUC的指标来validate-----------------------------
        # 添加了精确度、召回率、Matthews相关系数和平衡准确率。
    model.eval()
    predict = np.array([])
    target = np.array([])
    outputs[outputs < 0.5] = 0
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
    f1 = f1_score(target, predict)
    auc = roc_auc_score(target, predict)
    precision = precision_score(target, predict)
    recall = recall_score(target, predict)
    mcc = matthews_corrcoef(target, predict)
    bal_acc = balanced_accuracy_score(target, predict)
    print('Accuracy: {}'.format(acc))
    print('F1 Score: {}'.format(f1))
    print('Loss:', loss/step)
    print('AUC: {}'.format(auc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('MCC: {}'.format(mcc))
    print('Balanced Accuracy: {}'.format(bal_acc))

    

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
                                #--------------------对不平衡数据集进行SMOTE-----
            # smote = SMOTE(random_state=42)
            # inputs, labels = inputs.cpu().numpy(), labels.cpu().numpy()  # 将张量复制到主机内存
            # inputs, labels = smote.fit_resample(inputs, labels)
            # inputs, labels = torch.from_numpy(inputs).to(device), torch.from_numpy(labels).to(device)  # 将数据重新转换为张量并移回GPU

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


            # Convert predict and target to PyTorch tensors
            predict = torch.from_numpy(predict).to(device).cpu().numpy()
            target = torch.from_numpy(target).to(device).cpu().numpy()


            # Calculate F1 score
            f1 = f1_score(target, predict)
            auc = roc_auc_score(target, predict)
            precision = precision_score(target, predict)
            recall = recall_score(target, predict)
            mcc = matthews_corrcoef(target, predict)
            bal_acc = balanced_accuracy_score(target, predict)

            # Append F1 score to the list
            f1_scores.append(f1)
            auc_scores.append(auc)
            losses.append(loss)
            precisions.append(precision)
            recalls.append(recall)
            mccs.append(mcc)
            balanced_accuracies.append(bal_acc)

        
        epoch_loss /= step
        acc = accuracy_score(target, predict)
        print('tran_Accuracy: {}'.format(acc))
        save_loss(10, epoch_loss)
        validation(model,criterion2,test_dataloaders,device)
        # val_loss, val_acc = validation(model, criterion2, test_dataloaders, device)
        # print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_acc}')
# Save model
        

epochs_plot = np.arange(1,step+1)
plt.figure(figsize=(20, 12))

plt.subplot(2, 4, 1)
plt.plot(epochs_plot, f1_scores, label='F1 Score', marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')

plt.subplot(2, 4, 2)
plt.plot(epochs_plot, auc_scores, label='AUC', marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('AUC')

plt.subplot(2, 4, 3)
losses_trans = np.array([loss.item() for loss in losses])
plt.plot(epochs_plot, losses_trans, label='Loss', marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(2, 4, 4)
plt.plot(epochs_plot, precisions, label='Precision', marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Precision')

plt.subplot(2, 4, 5)
plt.plot(epochs_plot, recalls, label='Recall', marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Recall')

plt.subplot(2, 4, 6)
plt.plot(epochs_plot, mccs, label='MCC', marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('MCC')

plt.subplot(2, 4, 7)
plt.plot(epochs_plot, balanced_accuracies, label='Balanced Accuracy', marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Balanced Accuracy')

plt.savefig('plot.png')

torch.save(model.state_dict(), 'weights10_%d.pth' % (epoch))


