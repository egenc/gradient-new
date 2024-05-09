# Import libraries
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from sklearn import datasets

from methods import GradientDescent, ProposedDescent, StochasticDescent

iris = datasets.load_iris()

X = iris.data
Y = iris.target

'''
# Uncomment this code to use the loaded csv instead of the iris dataset
# Slice features and target from the advertising dataset
X = df[['TV', 'Radio', 'Newspaper']]
Y = df['Sales']

# Normalize target and features
Y = np.array((Y - Y.mean()) / Y.std())
X = X.apply(lambda rec: (rec - rec.mean()) / rec.std(), axis=0)
'''

LR_LIST = [0.00001, 0.01, 0.005, 0.001]
ITER_LIST = [1000, 10_000, 50_000]

JUMP_SIZES = [2, 3, 4, 5, 8, 10]

# Random initialization for bias and theta
def initialize(dim):
    b = random.random()
    theta = np.random.rand(dim)
    return b, theta


for it in ITER_LIST:
    for lr in LR_LIST:
        time_results = {}
        for jump_size in JUMP_SIZES:
            # Define the models to run
            models = {"stochastic_gradient_descent": StochasticDescent,
                    "gradient": GradientDescent,
                    "proposed": ProposedDescent}

            # Define the model parameters
            bias, weig = initialize(X.shape[1])
            params = {"iterations": it, "lr": lr}

            # Run the models and store the results and execution times in a dictionary
            results = {}
            times_dict = {}
            for model_name, model_class in models.items():
                if model_name == "proposed":
                    params["jump_size"] = jump_size
                    times = f"{model_name}_jump{jump_size}_time_elapsed"
                    result_name = f"{model_name}_jump{jump_size}_costs"
                else:
                    times = f"{model_name}_time_elapsed"
                    result_name = f"{model_name}_costs"
                start = time.time()
                model = model_class(weig, bias)
                costs, weights = model.fit(X, Y, **params)
                end = time.time()
                elapsed_time = end - start
                times_dict[times] = elapsed_time
                results[result_name] = costs

        series_dict = {}
        for k, v in results.items():
            series_dict[k] = pd.Series(v)

        df = pd.DataFrame(series_dict)

        df.to_csv(
            f"C:/Users/ERDG/Documents/repos/results/iris_lr{lr}_iter{it}.csv", index=False)
        times_df = pd.DataFrame.from_dict(
            times_dict, orient="index", columns=["time_elapsed"])
        times_df.to_csv(
            f"C:/Users/ERDG/Documents/repos/results/TIMES_iris_lr{lr}_iter{it}.csv", index_label="model_name")
