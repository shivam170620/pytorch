#%%
import torch
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
from sklearn.metrics import accuracy_score
from torchvision.datasets import ImageFolder


# %%
os.getcwd()
# %%
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.CenterCrop((64, 64)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
# %%
batch_size = 5
train_data = ImageFolder(root='data/train', transform=transform) 
test_data = ImageFolder(root='data/test', transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
classes = ['Negative', 'Positive']

img, label = train_data[0]
print(img.shape)
# %%
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# %%
# stride = 1 , padding = 0 , Kernal size = 3 , input size = 224
# out = (W - F + 2P) / S + 1

# Image Type	Channels
# Grayscale image	1
# RGB image	3

# Output Channels (out_channels)

# This is the number of filters the layer learns.

# Each filter produces one feature map.
#%%
class ImageClassificationNN(nn.Module):
    def __init__(self):
      super().__init__()
      self.Conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=0)
    #   input channels = 1
    #     output channels = 6
    #     kernel size = 3
    #     stride = 1
    #     padding = 0 , output of the first conv layer will be (64 - 3 + 2*0) / 1 + 1 = 62 , so the output of the first conv layer will be (6, 62, 62)
      self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output of the first maxpool layer will be (6, 31, 31)
      self.Conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0) # output of the second conv layer will be (16, 29, 29)
      self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output of the second maxpool layer will be (16, 14, 14)
      self.Flatten = nn.Flatten() # Channels × Height × Width , channel is 16, weight and height is 14 for both 
      self.Linear1 = nn.Linear(16*14*14, 128)
      self.Linear2 = nn.Linear(128, 32)
      self.Linear3 = nn.Linear(32, 1)
      self.ReLU = nn.ReLU()
      self.Sigmoid = nn.Sigmoid()
    def forward(self,x):

        x = self.Conv1(x)
        x = self.ReLU(x)
        x = self.maxpool1(x)
        x = self.Conv2(x)
        x = self.ReLU(x)
        x = self.maxpool2(x)
        x = self.Flatten(x)
        x = self.Linear1(x)
        x = self.ReLU(x)
        x = self.Linear2(x)
        x = self.ReLU(x)
        x = self.Linear3(x)
        x = self.Sigmoid(x)
        return x
# %%
model = ImageClassificationNN()
print(model)
# %%
loss_function = nn.BCELoss() # Binary Cross Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 10
losses=[]
# %%
for epoch in range(epochs):
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        # if labels are in shape of [5,] we need to reshape it to [5, 1] for the loss function from [0,1,0,1,0] to [[0],[1],[0],[1],[0]]
        outputs = model(inputs)
        loss = loss_function(outputs, labels.reshape(-1, 1).float())
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
# %%
y_test=[]
y_test_hat=[]

for i, data in enumerate(test_loader, 0):
    inputs, labels = data
    with torch.no_grad():
        y_test_hat_temp = model(inputs).round()
    y_test_hat.extend(y_test_hat_temp.numpy())
    y_test.extend(labels.numpy())

acc=accuracy_score(y_test, y_test_hat)
print(f'Accuracy: {acc*100:.2f} %')
# %%
