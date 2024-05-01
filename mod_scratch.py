# Import libraries
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from sklearn import datasets

# Load data and show the first rows
df = pd.read_csv("Advertising.csv")
df.head()

iris = datasets.load_iris()
# Slice data (only the first three features) and target
X = iris.data
Y = iris.target
# Print data shape
print(X.shape, Y.shape)

'''
# Uncomment this code to use the loaded csv instead of the iris dataset
# Slice features and target from the advertising dataset
X = df[['TV', 'Radio', 'Newspaper']]
Y = df['Sales']

# Normalize target and features
Y = np.array((Y - Y.mean()) / Y.std())
X = X.apply(lambda rec: (rec - rec.mean()) / rec.std(), axis=0)
'''

# Random initialization for bias and theta
def initialize(dim):
    b = random.random()
    theta = np.random.rand(dim)
    return b, theta

# Prediction function
def predict_Y(b, theta, X):
    return b + np.dot(X, theta)

# Cost function
def get_cost(Y, Y_hat):
    Y_resd = Y - Y_hat
    return np.sum(np.dot(Y_resd.T, Y_resd)) / len(Y - Y_resd)

# Gradient descent step
def update_theta(x, y, y_hat, b_0, theta_o, learning_rate):
    db = (np.sum(y_hat - y) * 2) / len(y)
    dw = (np.dot((y_hat - y), x) * 2) / len(y)
    b_1 = b_0 - learning_rate * db
    theta_1 = theta_o - learning_rate * dw
    return b_1, theta_1, db, dw

# Modified gradient descent step
def update_theta_modified(x, y, y_hat, b_0, theta_o, learning_rate, wpr, bpr, k):
    db = (np.sum(y_hat - y) * 2) / len(y)
    dw = (np.dot((y_hat - y), x) * 2) / len(y)
    b_1 = b_0 - learning_rate * (k * db + (1 - k) * bpr)
    theta_1 = theta_o - learning_rate * (k * dw + (1 - k) * wpr)
    return b_1, theta_1, db, dw

# Stochastic gradient descent step
def update_theta_stochastic(x, y, b_0, theta_o, learning_rate):
    rand_idx = random.choice(range(len(y)))
    x_i = x[rand_idx]
    y_i = y[rand_idx]
    db = 2 * (b_0 + np.dot(theta_o, x_i) - y_i)
    dw = x_i * db
    b_1 = b_0 - learning_rate * db
    theta_1 = theta_o - learning_rate * dw
    return b_1, theta_1

# Stochastic gradient descent function
def stochastic_gradient_descent(iterations, b, theta, lr):
    
    costs = []
    weights = []
    
    for i in range(iterations):
        # Perform a single update using the stochastic gradient descent algorithm
        b, theta = update_theta_stochastic(X, Y, b, theta, lr)
        
        # Calculate cost and store weights
        Y_hat = predict_Y(b, theta, X)
        cost = get_cost(Y, Y_hat)
        costs.append(cost)
        weights.append(theta)
        
    weights = np.array(weights).T
    
    return costs, weights

# Modified gradient descent function
def proposed_descent(iterations, b, theta, lr):
    
    costs = []
    weights = []
    md_list = [None] * iterations
    bd_list = [None] * iterations

    for i in range(iterations):
        if i == 0:
            Y_hat = predict_Y(b, theta, X)
            costs.append(get_cost(Y, Y_hat))
            weights.append(theta)
            b, theta, db0, dw0 = update_theta(X, Y, Y_hat, b, theta, lr)
            md_list[i] = dw0
            bd_list[i] = db0
        else:
            Y_hat = predict_Y(b, theta, X)
            costs.append(get_cost(Y, Y_hat))
            weights.append(theta)
            b, theta, db0, dw0 = update_theta_modified(X, Y, Y_hat, b, theta, lr, md_list[i - 1], bd_list[i - 1], 0.5)
            md_list[i] = dw0
            bd_list[i] = db0
                       
    weights = np.array(weights).T
    
    return costs, weights

# gradient descent function
def gradient_descent(iterations, b, theta, lr):

    costs = []
    weights = []
    db_list = []
    dw_list = []

    for i in range(iterations):
        Y_hat = predict_Y(b, theta, X)
        costs.append(get_cost(Y, Y_hat))
        weights.append(theta)
        b, theta, db, dw = update_theta(X, Y, Y_hat, b, theta, lr)
        db_list.append(db)
        dw_list.append(dw)
    weights = np.array(weights).T

    return costs, weights, db_list, dw_list

# Define the number of iterations, bias, weights, and learning rate
it = 45_000
bias, weig = initialize(X.shape[1])
lr = 0.01

times_dict = {}

# Measure the execution time of the modified function
start = time.time()
proposed_costs, proposed_weights = proposed_descent(it, bias, weig, lr)
end = time.time()
times_dict["proposed_time_elapsed"] = end - start

# Calculate costs, biases, and weights for the gradient descent function
start = time.time()
gradient_costs, gradient_weights, db, dw = gradient_descent(it, bias, weig, lr)
end = time.time()
times_dict["gradient_time_elapsed"] = end - start

# Measure the execution time of the stochastic_gradient_descent function
start = time.time()
stochastic_gradient_descent_costs, stochastic_gradient_descent_weights = stochastic_gradient_descent(it, bias, weig, lr)
end = time.time()
stochastic_time_elapsed = end - start
times_dict["stochastic_time_elapsed"] = stochastic_time_elapsed

## Getting fastest algo
fastest_algo = min(times_dict.items(), key=lambda x: x[1])
print(times_dict)

costs_dict = {"stochastic_gradient_descent_costs":stochastic_gradient_descent_costs[-1], 
              "gradient_costs":gradient_costs[-1],
              "proposed_costs":proposed_costs[-1]}

lowest_cost_algo = min(times_dict.items(), key=lambda x: x[1])
print(costs_dict)

# Create figure with two subplots in one row
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Plot modified function in the first subplot
COLOR = 'black'
ax1.plot(range(len(proposed_costs)), proposed_costs, color=COLOR)
ax1.scatter(range(len(proposed_costs)), proposed_costs, marker='o', color=COLOR)
ax1.set_xlabel('Weights')
ax1.set_ylabel('Modified results', color=COLOR)
ax1.tick_params(axis='y', labelcolor=COLOR)

# Plot gradient descent function in the second subplot
COLOR = 'tab:blue'
ax2.plot(range(len(gradient_costs)), gradient_costs, color=COLOR)
ax2.scatter(range(len(gradient_costs)), gradient_costs, marker='o', color=COLOR)
ax2.set_xlabel('Weights')
ax2.set_ylabel('gradient descent gradients', color=COLOR)
ax2.set_title('Values of modified and gradient descent over iterations')
ax2.tick_params(axis='y', labelcolor=COLOR)

# Plot stochastic gradient descent function in the second subplot
COLOR = 'tab:blue'
ax3.plot(range(len(stochastic_gradient_descent_costs)), stochastic_gradient_descent_costs, color=COLOR)
ax3.scatter(range(len(stochastic_gradient_descent_costs)), stochastic_gradient_descent_costs, marker='o', color=COLOR)
ax3.set_xlabel('Weights')
ax3.set_ylabel('stochastic gradient descent gradients', color=COLOR)
ax3.set_title('All Algos')
ax3.tick_params(axis='y', labelcolor=COLOR)

# Show the plots
plt.show()