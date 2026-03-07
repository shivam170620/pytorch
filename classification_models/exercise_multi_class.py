# full_data = pd.read_csv('/kaggle/input/faults.csv') 
# Solve the classification problem with the dataset 

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

#%%
path = kagglehub.dataset_download("shrutimechlearn/steel-plate-fault")
df = pd.read_csv(os.path.join(path, "faults.csv"))
print("Rows :", df.shape[0], "Columns :", df.shape[1])



# %%
# Exploratory Data Analysis 
df.head()
# %%
# how many unique classes are there in the target variable 
print(df['target'].unique())
df.columns
# %%
# Have the count, mean, std, min ,max , 25% , 50% , 75% of the data
print(df.describe())
# %%

## Visualise the data
sns.countplot(x='target', data=df, palette=sns.color_palette('Set2'))
plot.show()
df['target'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('Set2'))
# %%full_data.hist(figsize=(15,15))
df.hist(figsize=(15,15))
plot.show()
df.plot(kind="density", layout=(6,5), 
             subplots=True,sharex=False, sharey=False, figsize=(15,15))
plot.show()
# %%
df.isnull().sum()
# %%
df['X_Maximum'].fillna(df['X_Maximum'].median(), inplace=True) # right skwed data, fill the na values with mean or median, if skewed then with median if normal then mean
df['Steel_Plate_Thickness'].fillna(df['Steel_Plate_Thickness'].median(), inplace=True)
df['Empty_Index'].fillna(df['Empty_Index'].mean(), inplace=True)  
df.isna().sum()
# %%
def draw_univariate_plot(dataset, rows, cols, plot_type):
    column_names=dataset.columns.values
    number_of_column=len(column_names)
    fig, axarr=plot.subplots(rows,cols, figsize=(30,35))

    counter=0
    
    for i in range(rows):
        for j in range(cols):

            if column_names[counter]=='target':
                break
            if 'violin' in plot_type:
                sns.violinplot(x='target', y=column_names[counter],data=dataset, ax=axarr[i][j])
            elif 'box'in plot_type :
                #sns.boxplot(x='target', y=column_names[counter],data=dataset, ax=axarr[i][j])
                sns.boxplot(x=None, y=column_names[counter],data=dataset, ax=axarr[i][j])

            counter += 1
            if counter==(number_of_column-1,):
                break
# %%
draw_univariate_plot(dataset=df, rows=7, cols=4,plot_type="box")

# %%
X = df.drop('target', axis=1)
y = df['target']
# %%
# Neural nets expect the target as to be numeric values, so convert the target into numeric values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y[:10])
# %%
# Now we have to convert the data into float32 and then into torch tensors
X = X.astype('float32')
y = y.astype('int64') # for multiclass classification we have to convert
# %%
#----------------------------- Building the dataset and dataloader -----------------------------
from torch.utils.data import Dataset, DataLoader
import torch 
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.metrics import accuracy_score
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Label Encoder converts the textual distict values into numeric values
# Standard Scaler standardises the data by making all the features have the same scale, range and distribution, it is important for the neural network to learn the patterns in the data effectively.
# It is important to standardise the data before training a neural network because it helps the model to converge faster and can lead to better performance. 
# Standardisation ensures that all features are on the same scale, which can prevent certain features from dominating the learning process and can help the model to learn more effectively from the data.
# %%
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# %%
num_classes = len(np.unique(y))
print(num_classes)
# %%
class SteelPlateDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.len
# %%
training_dataset = SteelPlateDataset(X_train, y_train)
testing_dataset = SteelPlateDataset(X_test, y_test)

training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)
testing_dataloader = DataLoader(testing_dataset, batch_size=32, shuffle=False)
# %%
# Import the dataloader and create the dataloader for training and testing dataset
class NeuralNet(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.lin1 = nn.Linear(num_features, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        x = self.softmax(x)
        return x
# %%
## How to identify the size of haiiden layers and no of layers in the neural network
# how we idenitfy is the size of dataset and no of rows if less then less no of hidden layers 

# say 28 features -> 64 neurons in the first hidden layer -> 32 neurons in the second hidden layer -> 7 output classes
# if bigger dataset then we can increase the no of hidden layers and no of neurons in the hidden layers
# say eg is 80000 rows then 28-> 128->64->32->7

## How to calalculate the no of paramsmeters in neural network or model which we are building 
# no of params total = sum over all layers ( no of input features * no of neurons in the layer + no of neurons in the layer )
# say 28-> 64 -> 32 -> 7
# no of params in the first hidden layer = 28*64 + 64 = 1856
# no of params in the second hidden layer = 64*32 + 32 = 2080
# no of params in the output layer = 32*7 + 7 = 231
# total no of params = 1856 + 2080 + 231 = 4167
# %%
print(X_train.shape[0], num_classes, X_train.shape[1])
input_size = X_train.shape[1]
output_size = num_classes
model = NeuralNet(num_features=input_size, num_classes=output_size)
print(model)
# %%
loss_function = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# %%
epochs = 500
losses = []
for epoch in range(epochs):
    for x_, y_ in training_dataloader:
        optimizer.zero_grad()
        y_hat = model(x_)
        loss = loss_function(y_hat, y_)
        loss.backward()
        optimizer.step()
    losses.append(float(loss.data.detach().numpy()))
    print(f"Epoch {epoch}, Loss: {losses[-1]}")
# %%
sns.lineplot(x=range(epochs), y=losses)
# %%
model.eval()
all_preds = []
with torch.no_grad():
    for x_, y_ in testing_dataloader:
        predicted_outputs = model(x_)
        _, y_hat = torch.max(predicted_outputs, dim=1)
        all_preds.extend(y_hat.tolist())
        
# %%
accuracy_score(y_test, all_preds)
# %%
