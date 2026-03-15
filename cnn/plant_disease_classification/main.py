#%%
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
import kagglehub
import os
sns.set()

KAGGLE_API_TOKEN="KGAT_778ba30fbe06aada4155ec480cbbc9e9"

# Download latest version
path = kagglehub.dataset_download("rashikrahmanpritom/plant-disease-recognition-dataset")

print("Path to dataset files:", path)
for root, dirs, files in os.walk(path):
    print(root, len(files))
# %%
#### Exploratory Data Analysis 
#   Class Distribution , Images Qualities are consistent , Image shape and size Uniform ,
# Pixel / Color Distribution , Background Bias — Hidden danger, Intra-class vs Inter-class Similarity

path
# %%
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from collections import Counter
from torchvision import datasets
# %%
dataset_path = path   # kagglehub path
train_path = os.path.join(dataset_path, "Train")
train_path = os.path.join(train_path, "Train")
valid_path = os.path.join(dataset_path, "Validation")
valid_path = os.path.join(valid_path, "Validation")
test_path = os.path.join(dataset_path, "Test")
test_path = os.path.join(test_path, "Test")
print(train_path, test_path, valid_path)
# %%
classes = os.listdir(train_path)
counts={}

for cls in classes:
    cls_path = os.path.join(train_path, cls)
    counts[cls] = len(os.listdir(cls_path))
# %%
plt.figure(figsize=(6,4))
plt.bar(counts.keys(), counts.values())
plt.title("Class Distribution - Training Set")
plt.xlabel("Classes")
plt.ylabel("Number of Images")
plt.show()

print(counts)
# %%
### Image Quality Consistency 
plt.figure(figsize=(12,6))

for i, cls in enumerate(classes):
    cls_path = os.path.join(train_path, cls)
    
    sample_imgs = random.sample(os.listdir(cls_path), 3)

    for j, img_name in enumerate(sample_imgs):
        img_path = os.path.join(cls_path, img_name)
        img = Image.open(img_path)

        plt.subplot(len(classes), 3, i*3+j+1)
        plt.imshow(img)
        plt.title(cls)
        plt.axis('off')

plt.tight_layout()
plt.show()

# %%
### Dimension of the Images from train data set
sizes = []

for cls in classes:
    cls_path = os.path.join(train_path, cls)

    for img_file in os.listdir(cls_path):
        img = Image.open(os.path.join(cls_path, img_file))
        sizes.append(img.size)

unique_sizes = Counter(sizes)

print("Top image sizes:")
print(unique_sizes.most_common(10))
# %%
widths = [s[0] for s in sizes]
heights = [s[1] for s in sizes]

plt.figure(figsize=(6,4))
plt.scatter(widths, heights)
plt.title("Image Size Distribution")
plt.xlabel("Width")
plt.ylabel("Height")
plt.show()


# Top image sizes:
# [((4000, 2672), 976), ((2592, 1728), 105), ((5184, 3456), 82), ((4000, 3000), 80),
 # ((4608, 3456), 63), ((4032, 3024), 15), ((2421, 2279), 1)]# 

 ## height, width , number of images of that height width 
# %%
## Pixel and color distribution 

mean_rgb = []
std_rgb = []

for cls in classes:
    cls_path = os.path.join(train_path, cls)

    for img_file in os.listdir(cls_path):
        img = Image.open(os.path.join(cls_path, img_file)).convert('RGB')
        img = np.array(img) / 255.0

        mean_rgb.append(img.mean(axis=(0,1)))
        std_rgb.append(img.std(axis=(0,1)))

mean_rgb = np.mean(mean_rgb, axis=0)
std_rgb = np.mean(std_rgb, axis=0)

print("Mean RGB:", mean_rgb)
print("Std RGB:", std_rgb)
# %%
fig, axes = plt.subplots(3, 4, figsize=(12,9))

for row, cls in enumerate(classes):
    cls_path = os.path.join(train_path, cls)

    imgs = random.sample(os.listdir(cls_path), 4)

    for col, img_name in enumerate(imgs):
        img = Image.open(os.path.join(cls_path, img_name))

        axes[row, col].imshow(img)
        axes[row, col].axis('off')

        if col == 0:
            axes[row, col].set_title(cls)

plt.tight_layout()
plt.show()
# %%
#### Image Preprocessing 

# 1. Resize the image to the same dimension 
# 2. Normalization and convert images into tensors 


## calculate the mean and std for rgb 
from PIL import Image 
classes =os.listdir(train_path)

means = []
stds = []

for cls in classes:
    image_cls_path = os.path.join(train_path, cls)
    for image in os.listdir(image_cls_path):
        img = Image.open(os.path.join(image_cls_path, image))
        img = np.array(img)/255.0
        means.append(img.mean(axis=(0,1)))
        stds.append(img.std(axis=(0,1)))

means = np.mean(means, axis=0)
stds = np.std(stds, axis = 0)
print(f"means : {means}, std : {stds}")
# %%
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
# %%
mean_rgb = [0.47171182, 0.58919345, 0.39722319]
std_rgb= [0.17471065, 0.15711372 , 0.18001845]
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean_rgb , std_rgb)
])
# %%
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean_rgb , std_rgb)
])
# %%
train_dataset = torchvision.datasets.ImageFolder(train_path, transform=train_transform)
test_dataset = torchvision.datasets.ImageFolder(test_path, transform=test_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
# %%
images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)

#%%
import matplotlib.pyplot as plt

def imshow(img):
    img = img.permute(1,2,0).numpy()
    img = std_rgb * img + mean_rgb
    img = np.clip(img, 0, 1)
    plt.imshow(img)

images, labels = next(iter(train_loader))

plt.figure(figsize=(12,8))

for i in range(8):
    plt.subplot(2,4,i+1)
    imshow(images[i])
    plt.title(train_dataset.classes[labels[i]])
    plt.axis('off')

plt.show()

#%%
num_classes = len(classes)
class PlantDiseaseClassificationNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 42, 3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(42*26*26, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x=self.conv1(x)
        x=self.maxpool(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.maxpool(x)
        x=self.relu(x)
        x=self.conv3(x)
        x=self.maxpool(x)
        x=self.relu(x)
        x=self.flatten(x)
        x=self.linear1(x)
        x=self.relu(x)
        x=self.linear2(x)
        x=self.relu(x)
        x=self.linear3(x)
        # x=self.relu(x)
        # self.softmax(x)

        return x
# %%
epochs = 5
model = PlantDiseaseClassificationNN(num_classes)
loss_fn = nn.CrossEntropyLoss() # use softmax internally cross ebtropy loss 
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# %%
for epoch in range(epochs):
    for idx, data in enumerate(train_loader):
        x_, y_ = data
        optimizer.zero_grad()
        y_pred = model(x_)
        loss = loss_fn(y_pred, y_)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}')
# %%
y_test = []
y_test_hat = []
for i, data in enumerate(test_loader, 0):
    inputs, y_test_temp = data
    with torch.no_grad():
        y_test_hat_temp = torch.argmax(model(inputs), dim=1)
    
    y_test.extend(y_test_temp.numpy())
    y_test_hat.extend(y_test_hat_temp.numpy())

# %%
acc = accuracy_score(y_test_hat, y_test)
print(f'Accuracy: {acc*100:.2f} %')
# %% confusion matrix
confusion_matrix(y_test_hat, y_test)
# %%