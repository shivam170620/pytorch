# #%%
# import torch 
# import torchvision
# import torchvision.transforms as transforms
# import torch.nn as nn
# import os
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, confusion_matrix
# import numpy as np
# from torch.utils.data import DataLoader

# # %%
# os.getcwd()
# # %%
# transform = transforms.Compose([
#     transforms.Resize((50, 50)),
#     transforms.Grayscale(num_output_channels=1),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5])
# ])
# # %%
# batch_size = 4 
# trainset = torchvision.datasets.ImageFolder(root='train', transform=transform)
# testset = torchvision.datasets.ImageFolder(root='test', transform=transform)
# train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
# # %%
# CLASSES = ['affenpinscher', 'akita', 'corgi']
# NUM_CLASSES = len(CLASSES)
# # %%
# class MultiImageClassificationNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=0) # 48
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 24
#         self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3) # 22
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 11
#         self.linear1 = nn.Linear(16 * 11 * 11 , 128) # channels * height * width , size = (w - k + 2 * p )/stride + 1
#         self.linear2 = nn.Linear(128, 64)
#         self.linear3 = nn.Linear(64, NUM_CLASSES)
#         self.flatten = nn.Flatten()
#         self.relu = nn.ReLU()
#         self.logSoftmax = nn.LogSoftmax()

#     def forward(self, x ):
#         x=self.conv1(x)
#         x=self.maxpool1(x)
#         x=self.relu(x)
#         x=self.conv2(x)
#         x=self.maxpool2(x)
#         x=self.relu(x)
#         x=self.flatten(x)
#         x=self.linear1(x)
#         x=self.relu(x)
#         x=self.linear2(x)
#         x=self.relu(x)
#         x=self.linear3(x)
#         x=self.relu(x)
   
#         return x


# # %%
# # input = torch.rand(1, 1, 50, 50) # BS, C, H, W
# model = MultiImageClassificationNN() 
# # %%
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# # %%
# num_epochs = 10
# for epoch in range(num_epochs):
#     for i, data in enumerate(train_loader):
#         x_,y_ = data
#         optimizer.zero_grad()
#         y_pred = model(x_)
#         loss = loss_fn(y_pred, y_)
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}')
# # %%
# y_test = []
# y_test_hat = []

# for i, data in enumerate(test_loader, 0):
#     x_, y_ = data
#     with torch.no_grad():
#         y_pred = model(x_)

#     y_test.extend(y_.numpy())
#     y_test_hat.extend(y_pred.numpy())
    
# # %%
# # %%
# acc = accuracy_score(y_test, np.argmax(y_test_hat, axis=1))
# print(f'Accuracy: {acc*100:.2f} %')
# # %% confusion matrix
# confusion_matrix(y_test, np.argmax(y_test_hat, axis=1))
# # %%


#%% packages
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
os.getcwd()

# %% transform and load data
transform = transforms.Compose(
    [transforms.Resize((50,50)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))])

batch_size = 4
trainset = torchvision.datasets.ImageFolder(root='train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.ImageFolder(root='test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

# %%
CLASSES = ['affenpinscher', 'akita', 'corgi']
NUM_CLASSES = len(CLASSES)
class ImageMulticlassClassificationNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)  
        self.pool = nn.MaxPool2d(2, 2)  
        self.conv2 = nn.Conv2d(6, 16, 3) 
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 11 * 11, 128) # out: (BS, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, NUM_CLASSES)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()
    
    def forward(self, x):
        x = self.conv1(x) # out: (BS, 6, 48, 48)
        x = self.relu(x)
        x = self.pool(x) # out: (BS, 6, 24, 24)
        x = self.conv2(x) # out: (BS, 16, 22, 22)
        x = self.relu(x)
        x = self.pool(x) # out: (BS, 16, 11, 11)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# input = torch.rand(1, 1, 50, 50) # BS, C, H, W
model = ImageMulticlassClassificationNet()      
# model(input).shape

# %% 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
# %% training
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)

        
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch}/{NUM_EPOCHS}, Loss: {loss.item():.4f}')


# %% test
y_test = []
y_test_hat = []
for i, data in enumerate(testloader, 0):
    inputs, y_test_temp = data
    with torch.no_grad():
        y_test_hat_temp = model(inputs).round()
    
    y_test.extend(y_test_temp.numpy())
    y_test_hat.extend(y_test_hat_temp.numpy())

# %%
acc = accuracy_score(y_test, np.argmax(y_test_hat, axis=1))
print(f'Accuracy: {acc*100:.2f} %')
# %% confusion matrix
confusion_matrix(y_test, np.argmax(y_test_hat, axis=1))
# %%