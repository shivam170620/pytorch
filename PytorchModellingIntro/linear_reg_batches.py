#%%

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn.modules import Module

#%%

cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
# Predicting the miles per gallons features
# Features -   Unnamed: 0   mpg  cyl   disp   hp  drat     wt   qsec  vs  am  gear  carb 
#  Unnamed: 0   mpg  cyl   disp   hp  drat     wt   qsec  vs  am  gear  carb
# 0          Mazda RX4  21.0    6  160.0  110  3.90  2.620  16.46   0   1     4     4
# 1      Mazda RX4 Wag  21.0    6  160.0  110  3.90  2.875  17.02   0   1     4     4
# 2         Datsun 710  22.8    4  108.0   93  3.85  2.320  18.61   1   1     4     1
# 3     Hornet 4 Drive  21.4    6  258.0  110  3.08  3.215  19.44   1   0     3     1
# 4  Hornet Sportabout  18.7    8  360.0  175  3.15  3.440  17.02   0   0     3     2
df = pd.read_csv(cars_file)
show_plot = True
if show_plot:
    sns.scatterplot(data=df, x="wt", y="mpg")
    sns.regplot(x='wt', y='mpg', data=df)
    plt.show()

X_list = df['wt']
X_np = np.array(X_list , dtype=np.float32).reshape(-1, 1) # reshaping will convert (n , ) list of n values into (n,1) matrix and 
# neural networks expect 2d array input 
y_list = df['mpg']
y_np = np.array(y_list, dtype=np.float32).reshape(-1, 1)
X = torch.from_numpy(X_np)
y = torch.from_numpy(y_np)

#%%

class LinearRgeressionTorch(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRgeressionTorch, self).__init__()  # It initilaises all the components of nn.Module
        self.linear = nn.Linear(input_size, output_size) 
       
    def forward(self, X):
        return self.linear.forward(X)
    
input_dim = 1
output_dim = 1 
model = LinearRgeressionTorch(input_dim, output_dim)

loss = nn.MSELoss()
lr = 0.02
epochs = 2000
optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)

#%%

## Model Training 
slope=[]
bias = []
losses=[]
for epoch in range(epochs):
    optimizer.zero_grad()
    # set gradients to zero 
    y_pred = model.forward(X)
    loss_value = loss(y_pred, y)
    loss_value.backward()
    optimizer.step()

    for name,param in model.named_parameters():
        print(name, param.data)
        if param.requires_grad:
            if name == 'linear.weight':
                slope.append(param.data.numpy()[0][0])
            if name == 'linear.bias':
                bias.append(param.data.numpy()[0])

    losses.append(loss_value.item())
    if (epoch % 100 == 0):
        print(f"Epoch {epoch}, Loss: {loss_value.item()}")

#%%
sns.scatterplot(x=range(epochs), y=losses)
#%% visualise the bias development
sns.lineplot(x=range(epochs), y=bias)
#%% 
sns.lineplot(x=range(epochs), y=slope)

# %% check the result
y_pred = model(X).data.numpy().reshape(-1)
sns.scatterplot(x=X_list, y=y_list)
sns.lineplot(x=X_list, y=y_pred, color='red')




# %%
