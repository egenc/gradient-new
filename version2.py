# Import libraries
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from sklearn import datasets
# Load data and show the first rows
# df = pd.read_csv("Advertising.csv")
# df.head()

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
def update_theta_proposed(x, y, y_hat, b_0, theta_o, learning_rate, wpr, bpr, k):
    m = len(y)
    dw = 2 * np.dot(x.T, y_hat - y) / m
    db = 2 * np.sum(y_hat - y) / m
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


# Proposed gradient descent function
def proposed_descent(iterations, b, theta, lr, jump_size):
    
    costs = []
    weights = []
    md_list = []
    bd_list = []

    Y_hat = predict_Y(b, theta, X)
    costs.append(get_cost(Y, Y_hat))
    weights.append(theta)
    b, theta, db0, dw0 = update_theta(X, Y, Y_hat, b, theta, lr)
    md_list.append(dw0)
    bd_list.append(db0)

    for i in range(1, iterations, jump_size):            
        Y_hat = predict_Y(b, theta, X)
        costs.append(get_cost(Y, Y_hat))
        weights.append(theta)
        b, theta, db0, dw0 = update_theta_proposed(X, Y, Y_hat, b, theta, lr, md_list[-1], bd_list[-1], 0.5)
        md_list.append(dw0)
        bd_list.append(db0)
                       
    weights = np.array(weights).T
    
    return costs, weights

# gradient descent function
def gradient_descent(iterations, b, theta, lr):

    costs = []
    weights = []

    for i in range(iterations):
        Y_hat = predict_Y(b, theta, X)
        costs.append(get_cost(Y, Y_hat))
        weights.append(theta)
        b, theta, db, dw = update_theta(X, Y, Y_hat, b, theta, lr)
 
    weights = np.array(weights).T

    return costs, weights

LR_LIST = [0.00001, 0.01, 0.005, 0.001]
ITER_LIST = [1000, 10_000, 50_000]
# Define the number of iterations, bias, weights, and learning rate
it = 1_000
lr = 0.001

time_results = {}


bias, weig = initialize(X.shape[1])

results = {}

times_dict = {}

# Measure the execution time of the modified function
start = time.time()
proposed_costs, proposed_weights = proposed_descent(it, bias, weig, lr, jump_size=3)
end = time.time()
times_dict["proposed_time_elapsed"] = end - start

# Calculate costs, biases, and weights for the gradient descent function
start = time.time()
gradient_costs, gradient_weights = gradient_descent(it, bias, weig, lr)
end = time.time()
times_dict["gradient_time_elapsed"] = end - start

# Measure the execution time of the stochastic_gradient_descent function
start = time.time()
stochastic_gradient_descent_costs, stochastic_gradient_descent_weights = stochastic_gradient_descent(it, bias, weig, lr)
end = time.time()
stochastic_time_elapsed = end - start
times_dict["stochastic_time_elapsed"] = stochastic_time_elapsed

## Getting fastest algo
# fastest_algo = min(times_dict.items(), key=lambda x: x[1])
# print("fastest_algo: ", fastest_algo)

results = {"stochastic_gradient_descent_costs":stochastic_gradient_descent_costs, 
              "gradient_costs":gradient_costs,
              "proposed_costs":proposed_costs}
# df = pd.DataFrame.from_dict(results)
# df.to_csv(f"results_iris_lr{lr}_iter{it}.csv", index=False)
latest_values = {key: value[-1] for key, value in results.items()}
print(latest_values)
# print cost lengths
for k,v in results.items():
    print(k, len(v))
# lowest_cost_algo = min(results.items(), key=lambda x: x[1])
# print("lowest_cost_algo: ", lowest_cost_algo)

print(f"GD: {len(gradient_costs)}\nSGD: {len(stochastic_gradient_descent_costs)}\nProposed: {len(proposed_costs)}")

# # Create figure with two subplots in one row
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# # Plot modified function in the first subplot
# COLOR = 'black'
# ax1.plot(range(len(proposed_costs)), proposed_costs, color=COLOR)
# ax1.scatter(range(len(proposed_costs)), proposed_costs, marker='o', color=COLOR)
# ax1.set_xlabel('Costs')
# ax1.set_ylabel('Modified results', color=COLOR)
# ax1.tick_params(axis='y', labelcolor=COLOR)

# # Plot gradient descent function in the second subplot
# COLOR = 'tab:blue'
# ax2.plot(range(len(gradient_costs)), gradient_costs, color=COLOR)
# ax2.scatter(range(len(gradient_costs)), gradient_costs, marker='o', color=COLOR)
# ax2.set_xlabel('Costs')
# ax2.set_ylabel('gradient descent gradients', color=COLOR)
# ax2.set_title('Values of modified and gradient descent over iterations')
# ax2.tick_params(axis='y', labelcolor=COLOR)

# # Plot stochastic gradient descent function in the second subplot
# COLOR = 'tab:blue'
# ax3.plot(range(len(stochastic_gradient_descent_costs)), stochastic_gradient_descent_costs, color=COLOR)
# ax3.scatter(range(len(stochastic_gradient_descent_costs)), stochastic_gradient_descent_costs, marker='o', color=COLOR)
# ax3.set_xlabel('Costs')
# ax3.set_ylabel('stochastic gradient descent gradients', color=COLOR)
# ax3.set_title('All Algos')
# ax3.tick_params(axis='y', labelcolor=COLOR)

# # Show the plots
# plt.show()