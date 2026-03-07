#%% packages
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import numpy as np
# %%
iris_dataset = load_iris()
print(iris_dataset)
# %%
x=iris_dataset.data
y=iris_dataset.target
# %%
# Do the split 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# %%
# Convert it into floast32 and torch ..
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# %%
# y values is multi class values say 0, 1 , 2 ... 
# %%
# Now build the Iris Dataset class 
class IrisDataset(Dataset):
    def __init__(self, X_train, y_train):
        # superinit calls the constructor of the parent class 
        super().__init__()
        self.X = torch.from_numpy(X_train)
        self.y = torch.from_numpy(y_train)
        # for multiclass y values of integers concert it into longtype tensors 
        self.y = self.y.type(torch.LongTensor)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.len
    


# %%
# Build multiclass neural network 
class MulticlassNeuralNetwork(nn.Module):

    def __init__(self, num_features, num_layers, num_classes):
        super().__init__()
        self.lin1 = nn.Linear(num_features, num_layers)
        self.lin2 = nn.Linear(num_layers, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.lin1(x)
        # this linear model learns shared features for 3 classes
        sig = torch.sigmoid(x)
        # This regression push the values in the range of the [0,1]
        x = self.lin2(sig)
        # This gives the output i.e 3 values ( class values )
        # x = self.log_softmax(x)
        # This provides the probability for the same 
        return x
# %%
# Hyperparams 
num_features = x.shape[1]
num_layers = 10
num_classes = len(np.unique(y))
# %%
# Run the model on the data
model = MulticlassNeuralNetwork(num_features=num_features, num_layers=num_layers, 
                                num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
lr = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
num_epochs = 400
train_data_loader = DataLoader(dataset=IrisDataset(X_train, y_train), batch_size=32)
# %%
losses = []
for epoch in range(num_epochs):
    for x_, y_ in train_data_loader:

        optimizer.zero_grad()
        y_hat_log = model(x_)
        loss = criterion(y_hat_log, y_)
        loss.backward()
        optimizer.step()

    losses.append(float(loss.data.detach().numpy()))

# Show losses through plots over epochs 
# %
# %%
sns.lineplot(x= range(len(losses)), y = losses)
X_test_torch = torch.from_numpy(X_test)
with torch.no_grad():
    y_test_hat_softmax = model(X_test_torch)
    y_test_hat = torch.max(y_test_hat_softmax.data, 1)

# Accuracy 
# %%
accuracy_score(y_test, y_test_hat.indices)
# %%
