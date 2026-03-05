# full_data = pd.read_csv('/kaggle/input/faults.csv') 
# Solve the classification problem with the dataset 

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

#%%
path = kagglehub.dataset_download("shrutimechlearn/steel-plate-fault")
df = pd.read_csv(os.path.join(path, "faults.csv"))




