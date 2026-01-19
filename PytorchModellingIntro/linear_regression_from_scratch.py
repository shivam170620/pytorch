import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

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
# print(df.head())

# Generate the plots
show_plot = False
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

# print(X, "wt")
# print(y, "mpg")

w = torch.rand(1, requires_grad=True, dtype=torch.float32)
b = torch.rand(1, requires_grad=True, dtype=torch.float32) 

num_epochs = 410
learning_rate = 0.001

for epoch in range(num_epochs):
    for i in range(len(X)):
        y_pred = w*X[i] + b 
        loss = (torch.mean((y_pred-y[i])**2))
        loss.backward() # calculate gradients 
        loss_value = loss.item()

        # update weigths and biases 
        with torch.no_grad():
            # condition will reset the gradients and will be 0, it will not have all other ietration gradients piled up
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
            w.grad.zero_() # make the gradient zero after itr
            b.grad.zero_()

    if epoch % 10 == 0: 
         print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# now we tuned our weights and biases, now check on the original X_list..
# we have minimised the loss st lets how good model is predicting

y_pred = (X*w + b).detach().numpy().flatten()
print(X_list.shape, X.shape, X_np.shape)
print(y_list.shape, y_pred.shape, y_np.shape)

# Whenever we do plots we must use a list numpy list ...
sns.scatterplot(x=X_list, y=y_list)
sns.lineplot(x=X_list, y=y_pred, color='red')

# Errors
# For error calc, loss calc and accrracy must use tensors...

y_pred_list = np.array(y_pred, dtype=np.float32).reshape(-1, 1)
y_pred_tensor = torch.from_numpy(y_pred_list)

mse = torch.mean((y_pred_tensor - y) ** 2)
rmse = torch.sqrt(mse)
mae = torch.mean(torch.abs(y_pred_tensor - y))
y_mean = torch.mean(y)
ss_tot = torch.sum((y - y_mean) ** 2)
ss_res = torch.sum((y - y_pred_tensor) ** 2)
r2 = 1 - ss_res / ss_tot

tolerance = 1.0  # mpg
accuracy = torch.mean((torch.abs(y_pred_tensor - y) < tolerance).float())

print("RMSE:", rmse.item())
print("R²:", r2.item())
print("accuracy", accuracy)