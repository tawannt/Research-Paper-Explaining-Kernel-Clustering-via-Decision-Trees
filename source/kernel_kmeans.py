# https://github.com/maxf14/explaining_kernel_clustering/blob/main/KernelkmeansFunctions.ipynb
import numpy as np
from sklearn.cluster import KMeans

rng = np.random.default_rng()

### Kernel k-means

### This is a naive implementation for the limited purposes of this paper.

def kerneldist(Kmat, y, x):
    
    seq = np.arange(Kmat.shape[0])
    n_clusters = np.arange(len(np.unique(y)))
    x_dist = np.zeros(len(n_clusters))
    
    for cluster in n_clusters:
        index = seq[y == cluster]
        if len(index)==0:
            x_dist[cluster] = 0
        else:
            x_dist[cluster] = (Kmat[x,x] + np.mean(Kmat[np.ix_(index,index)])-2*np.mean(Kmat[np.ix_(index,[x])]))
    
    return x_dist

def kernelkmeanscost(Kmat,y):
    
    seq = np.arange(Kmat.shape[0])
    n_clusters = np.arange(len(np.unique(y)))
    costs = np.zeros(len(n_clusters))
    
    for cluster in n_clusters:
        index = seq[y == cluster]
        if len(index)==0:
            costs[cluster]=0
        else:
            costs[cluster] = (np.sum(np.diag(Kmat[np.ix_(index,index)]))-(1/len(index))*np.sum(Kmat[np.ix_(index,index)]))

    return(costs)

def kernelkmeans(Kmat, n_clusters, algo, n_init = 10, n_iter = 100, silent = True):
    
    n_data = Kmat.shape[0]
    best_y = np.zeros(n_data)
    
    if algo == 'kmeans':
        Phi = Kmat
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
        kmeans.fit(Phi)
        best_y = kmeans.predict(Phi)
        centers = kmeans.cluster_centers_
        
        return(best_y, centers)
        
    elif algo=='kernelkmeans':
        best_cost = float('inf')
        
        for t in np.arange(n_init):
            ### initialize clusters
            if silent == False:
                print('Initialization #',t)
            y = rng.choice(n_clusters, n_data)
            converged = False
            it = 0
            
            while converged == False:
                ### assign to closest center
                c = np.arange(n_data)
                for x in range(n_data):
                    c[x] = np.argmin(kerneldist(Kmat,y,x))
                if np.array_equal(c,y)==True or it == n_iter:
                    converged = True
                    if silent == False:
                        print('Converged at', it)
                else:
                    y = c
                    it = it + 1
                    
            cost_t = np.sum(kernelkmeanscost(Kmat, y))
            
            if cost_t < best_cost:
                best_y = y
                best_cost = cost_t
        
        return(best_y)