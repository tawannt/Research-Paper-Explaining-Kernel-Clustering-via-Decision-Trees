# https://github.com/maxf14/explaining_kernel_clustering/blob/main/ExplainabilityFunctions.ipynb
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

### IMM Function

def find_cut(X, y, index_u, clusters_u, centers, check_all_cuts, silent):
  
    # X,y are the data and the lables
    # index_u is the array of indices at this node (n_u)
    # clusters_u is a vector of admissible cluster labels at the node (k_u)
    # centers is a matrix of all centers (k,d)

    # Data at this node
    X_u = X[index_u,:]
    y_u = y[index_u]
    
    #print(X_u)
    #print('Remaining Cluster Centers:', centers[clusters_u,:])
    
    # Find index of all points that still have their center available (in clusters)
    index_good = np.where(np.isin(y_u, clusters_u))[0]
    
    # These are these points
    X_relevant = X_u[index_good,:]
    
    #... and these are their clusters
    y_relevant = y_u[index_good]
    y_relevant = y_relevant.astype(int)
    
    if silent == False:
        print('Remaining relevant data:')
        print(X_relevant)
        print('Labels:')
        print(y_relevant)
        
  
    mistakes = float('inf')

    for j in np.arange(np.shape(X_relevant)[1]): # iterate over all coordinates
        
        z = centers[clusters_u,j] # projected centers
        
        if len(np.unique(z))==1:
            mistakes_temp = mistakes # if all projections are identical, don't cut
        else:
            sorted_z = np.sort(z)
            
            if check_all_cuts == False:
                
                thetas = (sorted_z[:-1] + sorted_z[1:])/2
            
            elif check_all_cuts == True:
                
                theta_index = np.where((X_relevant[:,j] > sorted_z[0]) & (X_relevant[:,j] < sorted_z[-1]))[0]
                thetas = np.sort(np.unique(X_relevant[theta_index,j]))
                thetas = (thetas[:-1] + thetas[1:])/2
            
            if silent == False:
                print('Check Coordinate',j)
                print('Projected Centers:',z)
                print('Thetas:', thetas)
            
            for theta in thetas:
                
                pointL_centerR = np.where((X_relevant[:,j] < theta) & (centers[y_relevant,j] >= theta))[0]
                pointR_centerL = np.where((X_relevant[:,j] >= theta) & (centers[y_relevant,j] < theta))[0]
                
                mistakes1 = len(pointL_centerR)
                mistakes2 = len(pointR_centerL)
                
                mistakes_temp = mistakes1 + mistakes2
                
                if silent == False:
                    print('Check Threshold', theta)
                    print((X_relevant[:,j] < theta) & (centers[y_relevant,j] >= theta))
                    print('Point left, center right:', pointL_centerR)
                    print((X_relevant[:,j] >= theta) & (centers[y_relevant,j] < theta))
                    print('Point right, center left:', pointR_centerL)
                    print('--->', mistakes_temp, 'Mistakes')
                
                if mistakes_temp < mistakes:
                    mistakes = mistakes_temp
                    best_cut = {'coordinate': j, 'threshold': theta, 'mistakes': mistakes}
    
    index_go_L = X_u[:,best_cut['coordinate']] <= best_cut['threshold']
    index_go_R = X_u[:,best_cut['coordinate']] > best_cut['threshold']
    
    best_cut['index_u_L'] = index_u[index_go_L]
    best_cut['index_u_R'] = index_u[index_go_R]
    
    best_cut['clusters_u_L'] = clusters_u[np.where(centers[clusters_u,best_cut['coordinate']] <= best_cut['threshold'])[0]]
    best_cut['clusters_u_R'] = clusters_u[np.where(centers[clusters_u,best_cut['coordinate']] > best_cut['threshold'])[0]]
    
    print('Cluster Label Partioning:', best_cut['clusters_u_L'], 'and', best_cut['clusters_u_R'])
    
    return(best_cut)

def do_cut(X, y, index_nodes, clusters_nodes, centers, threshold_cuts, check_all_cuts, silent):
    
    ### X,y are simply the data and the labels
    ### index_nodes is a list of arrays, each containing the index of points at a node
    
    print('+++ New Cut +++')
    mistakes = float('inf')
    n_nodes = len(index_nodes)
    n_leaves = 0
    
    for u in np.arange(n_nodes):
        
        print('--- Check node', u, '---')
        index_u = index_nodes[u]
        clusters_u = clusters_nodes[u]
        print('Node', u ,'contains cluster labels', clusters_u)
    
        if len(clusters_u)==1:
            print('... This is already a Leaf')
            n_leaves = n_leaves + 1
        else:
            best_cut_u = find_cut(X, y, index_u, clusters_u, centers, check_all_cuts, silent)
            mistakes_u = best_cut_u['mistakes']
            print('Mistakes:', mistakes_u)
        
            if mistakes_u < mistakes:
                best_u = u
                best_cut = best_cut_u
                mistakes = mistakes_u
    
    if n_leaves == n_nodes:
        
        print('+++ IMM has finished +++')
        
        return('onlyleaves')
    
    elif n_leaves < n_nodes:    
        
        print('Cut node', best_u, 
              'at Coordinate', best_cut['coordinate'], 
              'Threshold', best_cut['threshold'],
              '---> Mistakes =', best_cut['mistakes'])

        index_update = index_nodes.copy()
        index_update[best_u] = best_cut['index_u_L']
        index_update.append(best_cut['index_u_R'])

        clusters_update = clusters_nodes.copy()
        clusters_update[best_u] = best_cut['clusters_u_L']
        clusters_update.append(best_cut['clusters_u_R'])
        
        threshold_cuts_update = threshold_cuts.copy()
        newrow = np.array([best_u, best_cut['coordinate'], best_cut['threshold']])
        threshold_cuts_update = np.vstack([threshold_cuts_update, newrow])
        
        return(index_update, clusters_update, threshold_cuts_update)
    
def imm(X, y, centers, check_all_cuts = True, silent = True):
    
    n_data = np.shape(X)[0]
    index_nodes = [np.arange(n_data)]
    clusters_nodes = [np.unique(y.astype(int))]
    threshold_cuts = np.zeros((0,3))
    
    converged = False
    
    while converged == False:
        
        cut = do_cut(X, y, index_nodes, clusters_nodes, centers, threshold_cuts, check_all_cuts, silent)
        
        if cut=='onlyleaves':
            converged = True
        else:
            index_nodes, clusters_nodes, threshold_cuts = cut
        
        #print('Threshold Cuts so far:', threshold_cuts)
    
    y_imm = np.zeros(n_data)
    
    for obs in np.arange(n_data):
        y_imm[obs] = np.where([obs in index_u for index_u in index_nodes])[0][0]
    
    return(y_imm, threshold_cuts)

run = False

if run == True:

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    rng = np.random.default_rng()

    n_clusters = 2
    n_data = 6
    sigma = 0.2
    cov = np.array([[sigma, 0], [0, sigma]])
    X = np.zeros((n_data,2))

    for k in range(0,n_clusters):
        mean = np.array([k, (1-k)**2])
        n_k = int(n_data/n_clusters)
        X[k*n_k + np.array(range(0,n_k)),] = np.random.default_rng().multivariate_normal(mean, cov, n_k)

    kmeans = KMeans(n_clusters=n_clusters, n_init=3)
    kmeans.fit(X)
    y_true = kmeans.predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=y_true, s=50, cmap='viridis')

    centers = kmeans.cluster_centers_

    y_imm, threshold_cuts = imm(X, y_true, centers, check_all_cuts = False, silent = False)

    print(y_imm)

def taylor_imm(X, y, gamma, features_per_dim, check_all_cuts):
    from scipy.special import loggamma
    # We use d Taylor approximations to the Gaussian kernel as features

    N,d = X.shape
    true_k = len(np.unique(y))

    Phi = np.zeros((N, features_per_dim*d))

    for i in np.arange(d):
        X_i = X[:,i] # project data to ith dimension
        for n in np.arange(N):
            for j in np.arange(features_per_dim):
                # Use log space for numerical stability
                if j == 0:
                    coeff = 1.0
                else:
                    # log(sqrt((2*gamma)^j / j!)) = 0.5 * (j*log(2*gamma) - log(j!))
                    log_coeff = 0.5 * (j * np.log(2*gamma) - loggamma(j + 1))
                    coeff = np.exp(log_coeff)
                Phi[n,i*features_per_dim+j] = coeff * (X_i[n]**j) * np.exp(-gamma*X_i[n]**2)

    #print(Phi.shape)

    taylor_centers = np.zeros((true_k,features_per_dim*d))
    
    for i in np.arange(true_k):
        Phi_i = Phi[np.where(y==i)[0],:]
        taylor_centers[i,:] = np.mean(Phi_i, axis=0)
    
    y_imm, threshold_cuts = imm(Phi, y, taylor_centers, check_all_cuts = check_all_cuts)
    
    return(y_imm, threshold_cuts)
    
def kernelmatrix_imm(X, y, gamma, kernel, check_all_cuts):

    # We use the d univariate kernel matrices as features
    
    N,d = X.shape
    true_k = len(np.unique(y))

    Phi = np.zeros((N, N*d))

    for i in np.arange(d):
        X_i = X[:,i] # project data to ith dimension
        X_i = np.reshape(X_i, (-1, 1))
        Phi[:,i*N:(i+1)*N] = pairwise_kernels(X_i, metric=kernel, gamma=gamma)

    Kmat_centers = np.zeros((true_k,N*d))

    for i in np.arange(true_k):
        Phi_i = Phi[np.where(y==i)[0],:]
        Kmat_centers[i,:] = np.mean(Phi_i, axis=0)

    y_imm, threshold_cuts = imm(Phi, y, Kmat_centers, check_all_cuts = check_all_cuts)
    
    return(y_imm, threshold_cuts)

