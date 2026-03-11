# https://github.com/maxf14/explaining_kernel_clustering/blob/main/Code%20for%20Experiments%20(Pathbased%2C%20Aggregation%2C%20Flame%2C%20Iris%2C%20Cancer).ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def load_pathbased():
    df = pd.read_csv('data/pathbased.arff', skiprows=13, sep=",", header=None)

    X = np.array(df)[:,0:3]
    y_true = X[:,2]
    X = X[:,[0,1]]
    y_true = y_true.astype(int) - 1

    return X, y_true

def load_aggregation():
    df = pd.read_csv('data/aggregation.arff', skiprows=12, sep=",", header=None)

    X = np.array(df)[:,0:3]
    y_true = X[:,2]
    y_true = y_true.astype(int) - 1
    X = X[:,[0,1]]

    return X, y_true

def load_flame():
    df = pd.read_csv('data/flame.arff', skiprows=10, sep=",", header=None)

    X = np.array(df)[:,0:3]
    y_true = X[:,2]
    X = X[:,[0,1]]
    y_true = y_true.astype(int) - 1

    return X, y_true

def load_iris():
    iris = datasets.load_iris()
    X = iris.data
    y_true = iris.target

    return X, y_true

def load_breast_cancer():
    X, y_true = datasets.load_breast_cancer(return_X_y=True)

    return X, y_true

def load_dataset(dataset):
    if dataset == "Pathbased":
        return load_pathbased()
    elif dataset == "Aggregation":
        return load_aggregation()
    elif dataset == "Flame":
        return load_flame()
    elif dataset == "Iris":
        return load_iris()
    elif dataset == "Cancer":
        return load_breast_cancer()
    
def plot_result(X, y_kmeans, y_kkm, y_expand, y_kmeans_imm, y_imm, y_exkmc):
    plt.subplot(2, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], s=50, c=y_kmeans)
    plt.title('K-means', fontsize=10)

    plt.subplot(2, 3, 2)
    plt.scatter(X[:, 0], X[:, 1], s=50, c=y_kkm)
    plt.title('Kernel k-means', fontsize=10)

    plt.subplot(2, 3, 3)
    plt.scatter(X[:, 0], X[:, 1], s=50, c=y_expand)
    plt.title('Kernel IMM expanded', fontsize=10)

    plt.subplot(2, 3, 4)
    plt.scatter(X[:, 0], X[:, 1], s=50, c=y_kmeans_imm)
    plt.title('IMM on k-means', fontsize=10)

    plt.subplot(2, 3, 5)
    plt.scatter(X[:, 0], X[:, 1], s=50, c=y_imm)
    plt.title('Kernel IMM', fontsize=10)

    plt.subplot(2, 3, 6)
    plt.scatter(X[:, 0], X[:, 1], s=50, c=y_exkmc)
    plt.title('Kernel ExKMC', fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_metric(price_values, rand_values):
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.bar(['Kernel IMM', 'Kernel Expand', 'Kernel ExKMC'], price_values, color=['green', 'cyan', 'purple'], width=0.5)
    plt.title('Price of explainability', fontsize=12)
    plt.ylabel('Price of explainability')
    plt.ylim(0.9, 1.1)

    plt.subplot(1, 2, 2)
    plt.bar(['k-means', 'IMM', 'Kernel k-means', 'Kernel IMM', 'Kernel ExKMC', 'Kernel Expand'], rand_values, color=['red', 'orange', 'blue', 'green', 'purple', 'cyan'], width=0.6)
    plt.title('Rand index')
    plt.ylabel('Rand index')
    plt.ylim(0, 1.1)

    plt.tight_layout()
    plt.show()