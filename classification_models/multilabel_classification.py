#
# %%
from sklearn.datasets import make_multilabel_classification
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %%
X, y =  make_multilabel_classification( n_samples=10000,
    n_features=10,
    n_classes=3,
    n_labels=2)
# %%
print(X.shape)
print(y.shape)
# %%
# Now we have to convert the data into float32 and then into torch tensors
X = torch.from_numpy(X.astype('float32'))
y = torch.from_numpy(y.astype('float32'))
# %%
class MultiLabelDataset(Dataset):
    def __init__(self, X , y):
        super().__init__()
        self.X = X
        self.y = y
        self.len = self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.len
# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
train_dataset = MultiLabelDataset(X_train, y_train)
test_dataset = MultiLabelDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# %%
class MultiLabelNN(nn.Module):
    def __init__(self, num_features, hidden_size , num_classes):
        super().__init__()
        self.linear1 = nn.Linear(num_features, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        # self.num_labels = num_labels

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x
# %%
input_dim = X.shape[1]
hidden_size = 32
num_classes = 3
num_labels = 2
model = MultiLabelNN(num_features=input_dim, hidden_size=hidden_size, num_classes=num_classes)
# %%
epochs = 100
loss_function = nn.BCELoss() # Binary Cross Entropy Loss with Logits 
# “Logits” = raw outputs of the neural network before sigmoid.
# sigmoid introduces independence of scores for each class, allowing the model to predict multiple classes simultaneously, thats why not using softmax here in model as it sums the probabilties to 1 and is used for mutually exclusive classes.
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# %%
slope=[]
bias = []
losses = []
for epoch in range(epochs):
    for j, (X_, y_) in enumerate(train_dataloader):
        optimizer.zero_grad()
        y_pred = model(X_)
        loss_value = loss_function(y_pred, y_)
        loss_value.backward()
        optimizer.step()
    losses.append(loss_value.item())
    print(f"Epoch {epoch}, Loss: {loss_value.item()}")

    for name, param in model.named_parameters():
        if param.requires_grad:
            if name == 'linear1.weight':
                slope.append(param.data.numpy()[0][0]) # inside 2d arry
            if name == 'linear1.bias':
                bias.append(param.data.numpy()[0])

# %%
sns.scatterplot(x=range(len(losses)), y=losses)
# %%
sns.lineplot(x=range(len(slope)), y=slope)
# %%
sns.lineplot(x=range(len(bias)), y=bias)
# %%
with torch.no_grad():
    y_pred = model(X_test).round()# we can use round() to convert the predicted probabilities into binary predictions (0 or 1) for each class. This is because the output of the sigmoid function is a probability between 0 and 1, and by rounding it, we can determine whether the model predicts the presence (1) or absence (0) of each class for each sample. Alternatively, you could also use a threshold (e.g., 0.5) to achieve the same result, like this: y_pred_binary = (y_pred > 0.5).float(). The choice between rounding and using a threshold depends on your specific requirements and the distribution of your data.
    # # now we. get the outputs for the classes as independent score for each class using sigmoid activation function, we can set a threshold to convert these scores into binary predictions (0 or 1) for each class.
    # print(y_pred)
    # y_pred_binary = (y_pred > 0.7).float() # threshold of 0.5 is commonly used for binary classification tasks, but you can adjust it based on the specific requirements of your problem and the distribution of your data.
    # print(y_pred_binary)
    # print(y_test)

# %%

y_pred_labels = y_pred.numpy()
y_true = y_test.numpy()
# %%
accuracy_score(y_true, y_pred_labels)
# %%
