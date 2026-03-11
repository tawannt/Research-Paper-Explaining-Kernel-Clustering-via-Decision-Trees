# https://github.com/maxf14/explaining_kernel_clustering/blob/main/Code%20for%20Experiments%20(Pathbased%2C%20Aggregation%2C%20Flame%2C%20Iris%2C%20Cancer).ipynb
import numpy as np

### Define kernel functions

def rbf(x,y,gamma):
    return(np.exp(-gamma*np.sum((x-y)**2)))

def laplace(x,y,gamma):
    return(np.exp(-gamma*np.sum(np.abs(x-y))))

def linear(x,y):
    return(np.dot(x,y))