import numpy as np
import matplotlib.pyplot as plt

# Calling basic_cleaning_and_split from the preprocessing notebook to get the data
from preprocessing import  df_basic_cleaning_and_split, array_standardise_data

# Activation function
def relu(z):
    return np.maximum(0, z)

# Derivative of the activation function (relu)
def drelu(a):
    return (a > 0).astype(float)

# Root mean squared error loss function
# We times by 0.5 to make the mse derivative simpler.
#  
def mse_loss(y_pred, y_true, n):
    # Root mean squared error (scalar)
    return 0.5*np.mean((y_pred - y_true)**2)

# Used for setting the random seed for reproducibility
def set_seed(seed=0):
    np.random.seed(seed)

def forward(X, W1, b1, w2, b2):
    z1 = X @ W1 + b1
    # Applying activation function (relu) to the hidden layer
    a1 = relu(z1)

    z2 = a1 @ w2 + b2
    y_hat = z2 # Explain why not applying activation function to the output layer (regression problem)
    # Cache is used to store temporary values
    cache = {"z1": z1, "a1": a1, "z2": z2, "y_hat": y_hat}  
    return y_hat, cache

# Actual backpropagation
def backprop(X, y, W1, b1, w2, b2):
    N = X.shape[0] # No. of rows (number of samples)
    y_hat, cache = forward(X, W1, b1, w2, b2)
    a1 = cache["a1"] # Hidden layer 

    dL_dyhat = (y_hat - y) / N
    dL_dz2 = dL_dyhat  # ReLu not used on output (CHECK correct)

    dw2 = a1.T @ dL_dz2
    db2 = np.sum(dL_dz2, axis=0)

    dL_da1 = dL_dz2 @ w2.T
    dL_dz1 = dL_da1 * drelu(a1)

    dW1 = X.T @ dL_dz1
    db1 = np.sum(dL_dz1, axis=0)

    return {"dW1": dW1, "db1": db1, "dw2": dw2, "db2": db2}

def train_NN(X, y, hidden=64, lr=0.001, iters=10000, seed=2):
    set_seed(seed)
    W1 = 0.5*np.random.randn(X.shape[1], hidden)
    b1 = np.zeros(hidden)
    w2 = 0.5*np.random.randn(hidden, 1)
    b2 = np.zeros(1)

    losses = []
    for t in range(iters):
        y_hat, _ = forward(X, W1, b1, w2, b2)
        L = mse_loss(y_hat, y, n=X.shape[0])
        losses.append(L)

        grads = backprop(X, y, W1, b1, w2, b2)
        W1 -= lr * grads["dW1"]
        b1 -= lr * grads["db1"]
        w2 -= lr * grads["dw2"]
        b2 -= lr * grads["db2"]

    return W1, b1, w2, b2, losses
