# Import libraries
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from sklearn import datasets

from methods import GradientDescent, ProposedDescent, StochasticDescent
# Load data and show the first rows
# df = pd.read_csv("Advertising.csv")
# df.head()

iris = datasets.load_iris()
# Slice data (only the first three features) and target
X = iris.data
Y = iris.target
# Print data shape
# print(X.shape, Y.shape)
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

LR_LIST = [0.00001, 0.01, 0.005, 0.001]
ITER_LIST = [1000, 10_000, 50_000]
# Define the number of iterations, bias, weights, and learning rate
it = 1_000
lr = 0.001

time_results = {}


bias, weig = initialize(X.shape[1])

results = {}

times_dict = {}

# # Calculate costs, biases, and weights for the gradient descent function
start = time.time()

PD = ProposedDescent(weig, bias)
proposed_costs_1, proposed_weights = PD.fit(X, Y, iterations=it, lr=lr, jump_size=2)


PD_2 = ProposedDescent(weig, bias)
proposed_costs_2, proposed_weights = PD_2.fit(X, Y, iterations=it, lr=lr, jump_size=3)
end = time.time()
times_dict["gradient_time_elapsed"] = end - start

# # Calculate costs, biases, and weights for the gradient descent function
start = time.time()

GD = GradientDescent(weig, bias)
gradient_costs, gradient_weights = GD.fit(X, Y, iterations=it, lr=lr)

end = time.time()
times_dict["gradient_time_elapsed"] = end - start

# Measure the execution time of the stochastic_gradient_descent function
start = time.time()
SD = StochasticDescent(weig, bias)
stochastic_gradient_descent_costs, stochastic_gradient_descent_weights = SD.fit(X, Y, iterations=it, lr=lr)
end = time.time()
stochastic_time_elapsed = end - start
times_dict["stochastic_time_elapsed"] = stochastic_time_elapsed

results = {"stochastic_gradient_descent_costs":stochastic_gradient_descent_costs, 
              "gradient_costs":gradient_costs,
              "proposed_costs_1":proposed_costs_1,
              "proposed_costs_2":proposed_costs_2}

series_dict = {}
for k, v in results.items():
    series_dict[k] = pd.Series(v)

# Create the DataFrame
df = pd.DataFrame(series_dict)

df.to_csv(f"C:/Users/ERDG/Documents/repos/results/iris_lr{lr}_iter{it}.csv", index=False)

# times_df = pd.DataFrame.from_dict(times_dict)
# times_df.to_csv(f"C:/Users/ERDG/Documents/repos/results/TIMES_iris_lr{lr}_iter{it}.csv", index=False)



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