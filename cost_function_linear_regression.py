# Implementation of cost function for linear regression
# Numpy - a populat library for scientific computing
# Matplotlib - a populat library for plotting data
# Cost function is a measure how well our model is predicting the target price of the house. 
import numpy  as np
import matplotlib.pyplot as plt
from utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
plt.style.use('./deeplearning.mplstyle')

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])

# f_wb is a prediction is calculated
# the difference between the target and the prediction is calculated and squared. 
# This is added to the total cost

def compute_cost(x,y,w,b):
    """
    Computes the cost function for linear regression.

    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training data
    m = x.shape[0]
    
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i])**2
        cost_sum = cost_sum + cost
    total_cost = (1/(2*m)) * cost_sum
    
    return total_cost

# The goal is to find a model, with parameters, which will accurately predict house values given an input 
# The cost is a measure of how accurate the model is on the training data.
plt_intuition(x_train,y_train)
plt.close('all') 
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)
soup_bowl()