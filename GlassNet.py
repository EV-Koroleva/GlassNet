import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision.transforms import transforms 
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from pathlib import Path

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import copy
from sklearn.model_selection import train_test_split
from efficientnet_pytorch import EfficientNet
import cv2
from torchmetrics import F1

from fastprogress import master_bar, progress_bar

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_params = {
    'num_epochs': 100, 
    'batch_size': 32,
    'learning_rate': 0.0001,
    'embedding_dim': 128, 
    'sound_duration' : 3000,
    'eta_min': 1e-5,
    't_max': 10,
    'sample_size':4
}


image_path='./img_align_celeba'
df_attr=pd.read_csv('./list_attr_celeba.csv')
df_attr.replace(-1,0,inplace=True)
# df_attr = df_attr.head(1000)


# df = pd.DataFrame()
# df['image_id'] = df_attr['image_id']
# df['Eyeglasses'] = df_attr['Eyeglasses']
# df_attr = df

# Class to get data in specific format and preprocessing.
class CelebDataset(Dataset):
    def __init__(self,df_1,image_path,transform=None,mode='train'):
        super().__init__()
        self.attr=df_1.drop(['image_id'],axis=1)
        self.path=image_path
        self.image_id=df_1['image_id']
        self.transform=transform
        self.mode=mode
    def __len__(self):
        return self.image_id.shape[0]
    def __getitem__(self,idx:int):
        image_name=self.image_id.iloc[idx]
        image=Image.open(os.path.join(image_path,image_name))
        attributes=np.asarray(self.attr['Eyeglasses'].iloc[idx].T,dtype=np.float32)
        # labels=self.attr.columns.tolist()
        # a = labels[15]
        if self.transform:
            image=self.transform(image)
        return image,attributes 

class GlassNet(nn.Module):
    def __init__(self):
        super(GlassNet, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        
        self.classifier_layer = nn.Sequential(
            nn.Linear(1280 , 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256 , 128),
            nn.Linear(128 , 1),
            nn.Sigmoid()
        )
        
    def forward(self, inputs):
        x = self.model.extract_features(inputs)

        # Pooling and final linear layer
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.model._dropout(x)
        x = self.classifier_layer(x)
        return x     

def get_metrics(preds, target):
    target = target.int()
    # sm = nn.Softmax(dim=0)
    # preds = sm(preds)
    f1 = F1().cuda()
    score = f1(preds, target)
    return score.item()

# Split Dataset into train,test and valid
train_df,valid_df=train_test_split(df_attr,test_size=0.2,shuffle=True,random_state=46)

# Apply Data augmentation and different type of transforms on train data.
train_transform=transforms.Compose([transforms.Resize((224,224)),transforms.RandomVerticalFlip(p=0.5),
                                    transforms.RandomHorizontalFlip(p=0.5),transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5063, 0.4258, 0.3832],std=[0.2644, 0.2436, 0.2397])])
# Apply Data augmentation and different type of transforms on test and validation data.
valid_transform=transforms.Compose([transforms.Resize((224,224)),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5063, 0.4258, 0.3832],std=[0.2644, 0.2436, 0.2397])])

# Create Dataset Object and DataLoader for train and validation set 
train_data=CelebDataset(train_df,image_path,train_transform)
train_loader=DataLoader(train_data,batch_size=model_params['batch_size'],shuffle=True)
valid_data=CelebDataset(valid_df,image_path,valid_transform)
valid_loader=DataLoader(valid_data,batch_size=model_params['batch_size'])


model = GlassNet()
model.cuda()
model.train()
criterion = nn.BCELoss().cuda()

optimizer = Adam(params=model.parameters(), lr=model_params['learning_rate'], amsgrad=False)
scheduler = CosineAnnealingLR(optimizer, T_max=model_params['t_max'], eta_min=model_params['eta_min'])

prev_score = 0.0

mb = master_bar(range(model_params['num_epochs']))
for epoch in mb:
    running_loss = []
    tmp_metrics =[]
    for image_batch,target_batch in progress_bar(train_loader):

        
        preds = model(image_batch.cuda())[:,0]
        target_batch = target_batch.cuda()
        loss = criterion(preds, target_batch)
        running_loss.append(loss.cpu().detach().numpy())

        tmp_metrics.append(get_metrics(preds,target_batch))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('train = {0}'.format(np.mean(np.array(tmp_metrics))))
    tmp_metrics =[]

    for image_batch,target_batch in progress_bar(valid_loader):
        
        preds = model(image_batch.cuda())[:,0]
        target_batch = target_batch.cuda()
        loss = criterion(preds, target_batch)
        running_loss.append(loss.cpu().detach().numpy())

        tmp_metrics.append(get_metrics(preds,target_batch))

    score = np.mean(np.array(tmp_metrics))
    print('valid = {0}'.format(score))
    if score > prev_score:
        torch.save(model.state_dict(),  './models/best_model.pt')
        prev_score = score
    

    if epoch % 10 == 0:
        torch.save(model.state_dict(),  './models/' + 'model'+ str(epoch) +'.pt')
    print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, model_params['num_epochs'], np.mean(running_loss)))
