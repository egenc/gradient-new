import random
import numpy as np


class LinearRegressor:
    def __init__(self, theta, b):
        """
        Initializes LinearRegressor class with slope (theta) and intercept (b).

        Args:
        - theta (numpy array): slope of the regression line
        - b (float): intercept of the regression line
        """
        self.b = b
        self.theta = theta
        self.costs = []
        self.weights = []

    def predict_Y(self, X, b, theta):
        """
        Predicts the y values based on the input X, slope, and intercept.

        Args:
        - X (numpy array): input array
        - b (float): intercept of the regression line
        - theta (numpy array): slope of the regression line

        Returns:
        - Y (numpy array): predicted values of the regression line
        """
        return b + np.dot(X, theta)

    def get_cost(self, Y, Y_hat):
        """
        Calculates the mean squared error between the predicted values and the actual values.

        Args:
        - Y (numpy array): actual values
        - Y_hat (numpy array): predicted values

        Returns:
        - cost(float): mean squared error between predicted and actual values
        """
        Y_resd = Y - Y_hat
        return np.sum(np.dot(Y_resd.T, Y_resd)) / len(Y - Y_resd)


class GradientDescent(LinearRegressor):

    def update_theta(self, x, y, y_hat, learning_rate):
        """
        Updates the slope and intercept values.

        Args:
        - x (numpy array): input array
        - y (numpy array): actual values
        - y_hat (numpy array): predicted values
        - learning_rate (float): determines the step size at each iteration

        Returns:
        - b_1 (float): updated intercept value
        - theta_1 (numpy array): updated slope value
        """
        db = (np.sum(y_hat - y) * 2) / len(y)
        dw = (np.dot((y_hat - y), x) * 2) / len(y)
        b_1 = self.b - learning_rate * db
        theta_1 = self.theta - learning_rate * dw
        return b_1, theta_1

    def fit(self, X, Y, iterations=1000, lr=0.001):
        """
        Fits the regression line based on the input and actual values using the gradient descent algorithm. Stores
        the cost and weight history during iterations.

        Args:
        - X (numpy array): input array
        - Y (numpy array): actual values
        - iterations (int): number of times to run the update
        - lr (float): determines the step size at each iteration

        Returns:
        - costs (list): cost history during iterations
        - weights (numpy array): weight history during iterations
        """
        for i in range(iterations):
            Y_hat = self.predict_Y(X, self.b, self.theta)
            self.costs.append(self.get_cost(Y, Y_hat))
            self.weights.append(self.theta)

            self.b, self.theta = self.update_theta(X, Y, Y_hat, lr)

        self.weights = np.array(self.weights).T

        return self.costs, self.weights


class ProposedDescent(LinearRegressor):

    def update_theta(self, x, y, y_hat, learning_rate):
        """
        Updates the slope and intercept values.

        Args:
        - x (numpy array): input array
        - y (numpy array): actual values
        - y_hat (numpy array): predicted values
        - learning_rate (float): determines the step size at each iteration

        Returns:
        - b_1 (float): updated intercept value
        - theta_1 (numpy array): updated slope value
        - db (float): gradient of the intercept
        - dw (numpy array): gradient of the slope
        """
        db = (np.sum(y_hat - y) * 2) / len(y)
        dw = (np.dot((y_hat - y), x) * 2) / len(y)
        b_1 = self.b - learning_rate * db
        theta_1 = self.theta - learning_rate * dw
        return b_1, theta_1, db, dw


    def update_theta_proposed(self, x, y, y_hat, b_0, theta_o, learning_rate, wpr, bpr, k):
        """
        Updates the slope and intercept values with the proposed gradient descent algorithm

        Args:
        - x (numpy array): input array
        - y (numpy array): actual values
        - y_hat (numpy array): predicted values
        - b_0 (float): the previous intercept value
        - theta_o (numpy array): the previous slope value
        - learning_rate (float): determines the step size at each iteration
        - wpr (numpy array): the previous slope gradient
        - bpr (float): the previous intercept gradient
        - k (float): momentum factor

        Returns:
        - b_1 (float): updated intercept value
        - theta_1 (numpy array): updated slope value
        - db (float): gradient of the intercept
        - dw (numpy array): gradient of the slope
        """
        m = len(y)
        dw = 2 * np.dot(x.T, y_hat - y) / m
        db = 2 * np.sum(y_hat - y) / m
        b_1 = b_0 - learning_rate * (k * db + (1 - k) * bpr)
        theta_1 = theta_o - learning_rate * (k * dw + (1 - k) * wpr)
        return b_1, theta_1, db, dw

    def fit(self, X, Y, iterations, lr, jump_size):
        """
        Fits the regression line based on the input and actual values using the proposed gradient descent algorithm.
        Stores the cost and weight history during iterations.

        Args:
        - X (numpy array): input array
        - Y (numpy array): actual values
        - iterations (int): number of times to run the update
        - lr (float): determines the step size at each iteration
        - jump_size (int): the number of updates to 'jump' before using the proposed gradient descent algorithm

        Returns:
        - costs (list): cost history during iterations
        - weights (numpy array): weight history during iterations
        """
        md_list = []
        bd_list = []

        Y_hat = self.predict_Y(X, self.b, self.theta)
        self.costs.append(self.get_cost(Y, Y_hat))
        self.weights.append(self.theta)

        self.b, self.theta, db0, dw0 = self.update_theta(X, Y, Y_hat, lr)

        md_list.append(dw0)
        bd_list.append(db0)

        for i in range(1, iterations, jump_size):

            Y_hat = self.predict_Y(X, self.b, self.theta)
            self.costs.append(self.get_cost(Y, Y_hat))
            self.weights.append(self.theta)

            self.b, self.theta, db0, dw0 = self.update_theta_proposed(
                X, Y, Y_hat, self.b, self.theta, lr, md_list[-1], bd_list[-1], 0.5)

            md_list.append(dw0)
            bd_list.append(db0)

        weights = np.array(self.weights).T

        return self.costs, self.weights


class StochasticDescent(LinearRegressor):
    def update_theta_stochastic(self, x, y, b_0, theta_o, learning_rate):
        """
        Updates the slope and intercept values using stochastic gradient descent

        Args:
        - x (numpy array): input array
        - y (numpy array): actual values
        - b_0 (float): the previous intercept value
        - theta_o (numpy array): the previous slope value
        - learning_rate (float): determines the step size at each iteration

        Returns:
        - b_1 (float): updated intercept value
        - theta_1 (numpy array): updated slope value
        """
        rand_idx = random.choice(range(len(y)))
        x_i = x[rand_idx]
        y_i = y[rand_idx]
        db = 2 * (b_0 + np.dot(theta_o, x_i) - y_i)
        dw = x_i * db
        b_1 = b_0 - learning_rate * db
        theta_1 = theta_o - learning_rate * dw
        return b_1, theta_1

    def fit(self, X, Y, iterations, lr):
        """
        Fits the regression line based on the input and actual values using stochastic gradient descent. Stores the 
        cost and weight history during iterations.

        Args:
        - X (numpy array): input array
        - Y (numpy array): actual values
        - iterations (int): number of times to run the update
        - lr (float): determines the step size at each iteration

        Returns:
        - costs (list): cost history during iterations
        - weights (numpy array): weight history during iterations
        """
        costs = []
        weights = []
        b = self.b
        theta = self.theta

        for i in range(iterations):
            b, theta = self.update_theta_stochastic(X, Y, b, theta, lr)
        
            Y_hat = self.predict_Y(X, b, theta)
            cost = self.get_cost(Y, Y_hat)
            self.costs.append(cost)
            self.weights.append(theta)

        weights = np.array(weights).T

        return self.costs, self.weights