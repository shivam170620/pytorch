# Flow of the writing Neural Networks 

1. Data Preparation
2. Separate Independent and Dependent Variables 
3. Removing columns which might not be needed
4. Train/Test Split
5. Scale the data 
6. Build Neural Network Class
7. Hyperparameter 
8. Model Instance and training on the dataset 
9. Check Lossess
10. Iterate over test data 
11. Calculate Accuracy 
12. Baseline Classifier 
13. Confusion Matrix 


## How Is Your Network Learning Correctly?

    Let’s trace one training iteration.

    Step 1: Forward Pass
    z1 = W1·X + b1
    a1 = sigmoid(z1)
    z2 = W2·a1 + b2
    a2 = sigmoid(z2)


    Interpretation:

    a2 = probability of class 1
    Values in (0, 1)

    Step 2: Loss Measures Error
    Although you didn’t explicitly code it, the implicit loss is:

    This loss answers:

    “How wrong is the predicted probability?”

    Step 3: Backpropagation
    dz2 = a2 - y

    Meaning:

    Positive → prediction too high
    Negative → prediction too low

    This error flows backward:
    dz1 = W2·dz2 ⊙ σ'(z1)

    Step 4: Gradient Descent Updates
    W := W - lr * gradient

    Each update:

    Reduces loss
    Improves prediction
    Moves decision boundary

    Step 5: Repeated Over Epochs

    Over many iterations:
    Loss ↓
    Accuracy ↑
    Weights converge

    ✅ That is learning