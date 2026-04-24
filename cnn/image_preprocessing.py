#%%

import torch
from torchvision import transforms
from PIL import Image # PIL is the Python Imaging Library by Fredrik Lundh and contributors.
from matplotlib import pyplot as plt # plotting library for the Python programming language and its numerical mathematics extension NumPy.
import numpy as np # NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
import torch.nn as nn # PyTorch's neural network module, which provides a set of classes


# %%
image = Image.open('cat.jpg')
image
# %%
image.size
#%%

# Preprocessing the image
preprocess_image = transforms.Compose([
    # resize , rotate, crop, grau scale, flip, etc, convert to tensor, normalize, etc
    transforms.Resize((300, 300)),
    transforms.RandomRotation(degrees=15), # Randomly rotate the image by a degree between -15 and 15
    transforms.CenterCrop((224, 224)), # Crop the image to a size of 224x224 pixels, centered on the image
    transforms.Grayscale(),
    transforms.RandomVerticalFlip(), # Randomly flip the image vertically with a probability of 0.5
    # transforms.ToTensor(), # Convert the image to a PyTorch tensor, which is a multi-dimensional array that can be used for deep learning models
])
# %%
image_tensor = preprocess_image(image)
# image_tensor.shape
# %%
image_tensor
# image
# %%
