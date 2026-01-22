#%%
import numpy as np
import pandas as pd
import graphlib
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

#%% data import
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
df = pd.read_csv(cars_file)
df.head()
# %%
X_list = df['wt']
y_list = df['mpg']
X_np = np.array(X_list, dtype=np.float32).reshape(-1, 1)
y_np = np.array(y_list, dtype=np.float32).reshape(-1, 1)
X = torch.from_numpy(X_np)
y = torch.from_numpy(y_np)
# %%

class LinearRegressionModel(nn.Module):
    def __init__(self,input_size, output_size, *args, **kwargs):
        super(LinearRegressionModel, self).__init__(*args, **kwargs)
        self.linear = nn.Linear(input_size, output_size) 

    def forward(self, X):
        return self.linear.forward(X)
    
class LinearRegressionDataLoader(Dataset):
    r"""An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`. Subclasses could also
    optionally implement :meth:`__getitems__`, for speedup batched samples
    loading. This method accepts list of indices of samples of batch and returns
    list of samples.

    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs an index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    """

    def __init__(self,X , y):
        super().__init__()
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.shape[0]
    
train_loader = DataLoader(dataset=LinearRegressionDataLoader(X, y), batch_size=2)


input_dim = 1
output_dim = 1
model = LinearRegressionModel(input_size=input_dim, output_size=output_dim)
model.train()


# %%

loss_function = nn.MSELoss()
learning_rate = 0.02
optimizer = torch.optim.SGD(params= model.parameters(), lr=learning_rate)

# %%
losses=[]
slope=[]
bias = []
epochs = 1000

for epoch in range(epochs):
    for idx , (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
        # make gradients as zero 

        y_pred = model.forward(X)
        loss_value = loss_function(y_pred, y)
        losses.append(loss_value.item())
        loss_value.backward() # we differtiate the

        # Now update the parameters
        optimizer.step()
        # print(model.named_parameters())
        for name, param in model.named_parameters(): # use named parameters model.named_parameters
            # print(name, param)
            # linear.weight Parameter containing:
            # tensor([[-5.1106]], requires_grad=True)
            # linear.bias Parameter containing:
            # tensor([36.8847], requires_grad=True)
            # linear.weight Parameter containing:
            # tensor([[-5.1884]], requires_grad=True)
            if param.requires_grad:
                if name == 'linear.weight':
                    slope.append(param.data.numpy()[0][0]) # inside 2d arry
                if name == 'linear.bias':
                    bias.append(param.data.numpy()[0])
    
    # print loss
    if (epoch % 100 == 0):
        print(f"Epoch {epoch}, Loss: {loss_value.item()}")

        
# %%

sns.scatterplot(x=range(len(losses)), y = losses)
sns.lineplot(x=range(epochs), y=slope)
sns.lineplot(x=range(epochs), y=bias)
# %%


# Ruuning again same script -----------------------------------------------------------------------------------

#%% packages
import graphlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
import seaborn as sns

#%% data import
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

#%% visualise the model
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)

#%% convert data to tensor
X_list = cars.wt.values
X_np = np.array(X_list, dtype=np.float32).reshape(-1,1)
y_list = cars.mpg.values
y_np = np.array(y_list, dtype=np.float32).reshape(-1,1)
X = torch.from_numpy(X_np)
y_true = torch.from_numpy(y_np)

#%% Dataset and Dataloader
class LinearRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(dataset = LinearRegressionDataset(X_np, y_np), batch_size=2)


#%%
class LinearRegressionTorch(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

input_dim = 1
output_dim = 1
model = LinearRegressionTorch(input_size=input_dim, output_size=output_dim)
model.train()

# %% Mean Squared Error
loss_fun = nn.MSELoss()

#%% Optimizer
learning_rate = 0.02
# test different values of too large 0.1 and too small 0.001
# best 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# check trainloader return
# for i, data in enumerate(train_loader):
#     print(data)

#%% perform training
losses = []
slope, bias = [], []
number_epochs = 1000
for epoch in range(number_epochs):
    for j, (X, y) in enumerate(train_loader):
        # optimization
        optimizer.zero_grad()

        # forward pass
        y_pred = model(X)

        # compute loss
        loss = loss_fun(y_pred, y)
        losses.append(loss.item())

        # backprop
        loss.backward()

        # update weights
        optimizer.step()
    
    # get parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name == 'linear.weight':
                slope.append(param.data.numpy()[0][0])
            if name == 'linear.bias':
                bias.append(param.data.numpy()[0])


    # store loss
    losses.append(float(loss.data))
    # print loss
    if (epoch % 100 == 0):
        print(f"Epoch {epoch}, Loss: {loss.data}")

    

# %% visualise model training
sns.scatterplot(x=range(len(losses)), y=losses)

#%% visualise the bias development
# sns.lineplot(x=range(number_epochs), y=bias)
#%% visualise the slope development
# sns.lineplot(x=range(number_epochs), y=slope)

sns.lineplot(x=X_list, y=y_pred, color='red')

# %% check the result
model.eval()
y_pred = [i[0] for i in model(torch.from_numpy(X_np)).data.numpy()]
y = [i[0] for i in y_true.data.numpy()]
sns.scatterplot(x=X_list, y=y)
sns.lineplot(x=X_list, y=y_pred, color='red')
# %%
