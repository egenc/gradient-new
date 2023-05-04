import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from sklearn import datasets

df=pd.read_csv("Advertising.csv")
df.head()

iris = datasets.load_iris()
X = iris.data[:, :3]  # we only take the first two features.
Y = iris.target

print(X.shape, Y.shape)
# X=df[['TV','Radio','Newspaper']]
# Y=df['Sales']

# Y=np.array((Y-Y.mean())/Y.std())

# X=X.apply(lambda rec:(rec-rec.mean())/rec.std(),axis=0)

def initialize(dim):
    b=random.random()
    theta=np.random.rand(dim)
    return b,theta

def predict_Y(b,theta,X):
    return b + np.dot(X,theta)

def get_cost(Y,Y_hat):
    Y_resd=Y-Y_hat
    return np.sum(np.dot(Y_resd.T,Y_resd))/len(Y-Y_resd)

def update_theta_old(x,y,y_hat,b_0,theta_o,learning_rate):
    db=(np.sum(y_hat-y)*2)/len(y)
    dw=(np.dot((y_hat-y),x)*2)/len(y)
    b_1=b_0-learning_rate*db
    theta_1=theta_o-learning_rate*dw
    return b_1,theta_1, db, dw

def update_theta(x,y,y_hat,b_0,theta_o,learning_rate):
    db=(np.sum(y_hat-y)*2)/len(y)
    dw=(np.dot((y_hat-y),x)*2)/len(y)
    b_1= b_0-learning_rate*db
    theta_1 = theta_o-learning_rate*dw
    return b_1,theta_1,db,dw

def update_theta_modified(x,y,y_hat,b_0,theta_o,learning_rate, wpr, bpr,k):
    db=(np.sum(y_hat-y)*2)/len(y)
    dw=(np.dot((y_hat-y),x)*2)/len(y)
    b_1= b_0 -learning_rate * (k*db + (1-k)*bpr)
    theta_1 = theta_o-learning_rate * (k*dw + (1-k)*wpr)
    return b_1,theta_1,db,dw

def main_modified(iterations,b,theta,lr):
    
    costs = []
    weights = []
    md_list = [None] * iterations
    bd_list = [None] * iterations

    for i in range(iterations):
        if i==0:
            Y_hat= predict_Y(b,theta,X)
            #print(get_cost(Y,Y_hat))
            costs.append(get_cost(Y,Y_hat))
            weights.append(theta)
            b,theta,db0, dw0 = update_theta(X,Y,Y_hat,b,theta,lr)
            md_list[i] = dw0
            bd_list[i] = db0
        else:
            Y_hat= predict_Y(b,theta,X)
            #print(get_cost(Y,Y_hat))
            costs.append(get_cost(Y,Y_hat))
            weights.append(theta)
            b,theta,db0, dw0 = update_theta_modified(X,Y,Y_hat,b,theta,lr, md_list[i-1],bd_list[i-1],0.5)
            md_list[i] = dw0
            bd_list[i] = db0
                       
    weights = np.array(weights).T
    
    return costs, weights

def main_regular(iterations,b,theta,lr):

    costs = []
    weights = []
    db_list = []
    dw_list = []


    for i in range(iterations):
        Y_hat= predict_Y(b,theta,X)
        #print(get_cost(Y,Y_hat))
        costs.append(get_cost(Y,Y_hat))
        weights.append(theta)
        b,theta, db, dw=update_theta_old(X,Y,Y_hat,b,theta,lr)
        db_list.append(db)
        dw_list.append(dw)
    weights = np.array(weights).T

    return costs, weights, db_list, dw_list

it = 200
bias,weig=initialize(3)
lr= 0.001

start = time.time()
mod_costs, mod_weights = main_modified(it,bias,weig,lr)
end = time.time()
mod_time_elapsed = end - start
print("mod_weights:", len(mod_weights))
print("mod_costs:", len(mod_costs))
start = time.time()
reg_costs, reg_weights, db, dw = main_regular(it,bias,weig,lr)
end = time.time()
org_time_elapsed = end - start

#mod_costs_001 = main_modified(it,bias,weig,0.01)
print("Original time - Modified time",org_time_elapsed-mod_time_elapsed)
print(mod_costs[-1], reg_costs[-1])
if(mod_costs[-1] < reg_costs[-1]):
    print('Better')
else:
    print('Fail')

fig, ax1 = plt.subplots()
COLOR='black'
# ax1.plot(mod_costs, color=COLOR)
# ax1.plot(mod_weights[0],mod_costs, color=COLOR)
# ax1.scatter(mod_weights[0],mod_costs, marker='o', color=COLOR)
# ax1.set_xlabel('Iterations'); ax1.set_ylabel('Modified_results', color=COLOR)
# ax1.tick_params(axis='y', labelcolor=COLOR)
# -----
COLOR='tab:blue'
ax2 = ax1.twinx()
# ax2.plot(reg_costs, color=COLOR)
ax2.plot(reg_weights[0],reg_costs, color=COLOR)
ax2.scatter(reg_weights[0],reg_costs, marker='o', color=COLOR)
ax2.set_title('Values of modified and original over iterations')
ax2.set_ylabel('original gradients', color=COLOR)
ax1.tick_params(axis='y', labelcolor=COLOR)

print(db)
plt.show()


