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
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(10),
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

    def forward(self, x):
        ""