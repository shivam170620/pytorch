import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

## Load the dataset
DATA_PATH = "datasets/heart.csv"
df = pd.read_csv(DATA_PATH)
print(df.head())

## Data Preprocessing seprate the dependent and independent variables
X = df.drop('target', axis=1).values
y = df['target'].values
print(X.shape, y.shape)

## Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# random state does the shuffling of data before splitting into train and test sets

## Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Prevents exploding gradients
# Faster convergence
# Stable learning 
# Puts each feature on the same scale like mean 0 and variance 1, which is important for gradient descent optimization.
# Example: Age (0-50), Income(10000 - 1000000) , Blood Pressure(40-200) have different scales but after scaling they will be on same scale.

# Input → Hidden Layer → Sigmoid → Output → Sigmoid

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, lr=0.01):
        # Initialize weights and biases
        self.w1=np.random.randn(input_size, hidden_size)
        self.b1=np.zeros((1, hidden_size))
        self.w2=np.random.randn(hidden_size, 1)
        self.b2=np.zeros((1, 1))
        self.lr=lr
        self.a1 = None
        self.a2 = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_serivative(self, z):
        return z * (1 - z)
    
    def forward(self, X):
        # print("Forward Pass")
        # print(self.w1.shape, X.shape, self.b1.shape)
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        # print(self.w2.shape, self.a1.shape, self.b2.shape)
        return self.a2

    def loss(self, y_true, y_pred):
        m = y_true.shape[0]
        y_true = y_true.reshape(m, 1)
        eps = 1e-8  # numerical stability
        loss = -np.mean(
            y_true * np.log(y_pred + eps) +
            (1 - y_true) * np.log(1 - y_pred + eps)
        )
        return loss

    
    def backward(self, X, y):
        """
        X: (m, n_features)
        y: (m, 1)
        """

        m = y.shape[0]
        y = y.reshape(m, 1)

        # print("Backward Pass")

        # -------- Output layer --------
        dz2 = self.a2 - y                      # (m, 1)
        # print("dz2:", dz2.shape)

        dw2 = (1/m) * np.dot(self.a1.T, dz2)  # (hidden, 1)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)  # (1, 1)
        # print("dw2:", dw2.shape, "db2:", db2.shape)

        # -------- Hidden layer --------
        dz1 = np.dot(dz2, self.w2.T) * self.sigmoid_serivative(self.a1)
        # dz1: (m, hidden)
        # print("dz1:", dz1.shape)

        dw1 = (1/m) * np.dot(X.T, dz1)         # (n_features, hidden)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)  # (1, hidden)
        # print("dw1:", dw1.shape, "db1:", db1.shape)

        # -------- Gradient descent update --------
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1


input_size = X_train.shape[1]
hidden_size = 16
epochs = 1000
nn = NeuralNetwork(input_size, hidden_size, lr=0.5)
losses = []

for epoch in range(epochs):
    y_pred = nn.forward(X_train)
    loss = nn.loss(y_train, y_pred)
    nn.backward(X_train, y_train)
    # update the weights and biases
    losses.append(loss)
    if epoch % 100 == 0:
        plt.plot(losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.show()
        print(f"Epoch {epoch}, Loss: {loss}")

## NOW EVALUATE THE MODEL ON TEST DATASET

y_test_pred = nn.forward(X_test)
y_test_pred = (y_test_pred > 0.5).astype(int).flatten()
accuracy = np.mean(y_test_pred == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


#### Baseline model using Logistic Regression for comparison
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_log_reg_pred = log_reg.predict(X_test)
log_reg_accuracy = np.mean(y_log_reg_pred == y_test)
print(f"Logistic Regression Test Accuracy: {log_reg_accuracy * 100:.2f}%")