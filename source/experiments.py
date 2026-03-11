# https://github.com/maxf14/explaining_kernel_clustering/blob/main/RunExperiments.ipynb
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.cluster import adjusted_rand_score
from kernel import rbf, laplace, linear
from kernel_kmeans import kernelkmeans, kernelkmeanscost
from kernel_imm import imm, taylor_imm, kernelmatrix_imm
from kernel_exkmc import exkmc_build_on_imm, exkmc_min_cost_at_node
from kernel_expand import expand_build_on_imm, expand_min_cost_at_node

### Find best kernel (with bandwidth) according to Rand Index

def get_hyperparam(X, y_true, gammas):
    
    true_k = len(np.unique(y_true))
    best_rand = 0
    
    for gamma in gammas:
        
        print('Test Gamma =', gamma)
        rands = np.zeros(2)

        Kmat = pairwise_kernels(X, metric=rbf, gamma=gamma)
        y = kernelkmeans(Kmat, true_k, algo='kernelkmeans', n_init=10, n_iter=100)
        rands[0] = adjusted_rand_score(y_true, y)
        
        Kmat = pairwise_kernels(X, metric=laplace, gamma=gamma)
        y = kernelkmeans(Kmat, true_k, algo='kernelkmeans', n_init=10, n_iter=100)
        rands[1] = adjusted_rand_score(y_true, y)
        
        best_i = np.argmax(rands)
        
        if rands[best_i]>best_rand:
            best_gamma = gamma
            best_rand = rands[best_i]
            best_kernel = best_i
            
    return(best_gamma, best_kernel)

def imm_experiments(X, y_true, gammas, n_init = 10):
    
    true_k = len(np.unique(y_true))
    
    print('--- First k-means')
    Kmat_lin = pairwise_kernels(X, metric=linear)

    y_kmeans, centers_kmeans = kernelkmeans(X, true_k, algo='kmeans', n_init = n_init, n_iter=200)
    rand_kmeans = adjusted_rand_score(y_kmeans, y_true)
    print('---> Rand Score of k-means:', rand_kmeans)

    y_kmeans_imm, kmeans_threshold_cuts = imm(X, y_kmeans, centers_kmeans, check_all_cuts = True)
    rand_imm = adjusted_rand_score(y_kmeans_imm, y_true)
    print('---> Rand Score of linear IMM:', rand_kmeans)
    
    print('--- Now to kernels. Find kernel and gamma ---')
    gamma, best_kernel = get_hyperparam(X, y_true, gammas)
    
    if best_kernel==0:
        kernelfunc = rbf
    elif best_kernel==1:
        kernelfunc = laplace
    
    print('We choose the', str(kernelfunc), 'kernel with gamma =', gamma)
    print('--- Run kernel k-means ---')
    Kmat = pairwise_kernels(X, metric=kernelfunc, gamma=gamma)
    y_kkm = kernelkmeans(Kmat, true_k, algo='kernelkmeans', n_init = n_init, n_iter=200)

    rand_kkm = adjusted_rand_score(y_true, y_kkm)
    print('---> Rand Score:', rand_kkm)
    
    # For the Gaussian kernel, we try both Taylor as well as kernel matrix features
    if best_kernel==0:
        
        print('Run Gaussian Taylor IMM on kernel k-means')
        y_taylor_imm_on_kkm, threshold_cuts_taylor = taylor_imm(X, y_kkm, gamma, 5, check_all_cuts = True)

        rand_taylor_imm_on_kkm = adjusted_rand_score(y_taylor_imm_on_kkm, y_true)
        price_taylor_imm_on_kkm = np.sum(kernelkmeanscost(Kmat, y_taylor_imm_on_kkm))/np.sum(kernelkmeanscost(Kmat, y_kkm))

        print('Run Gaussian kernel matrix IMM on kernel k-means')
        y_kmat_imm_on_kkm, threshold_cuts_kmat = kernelmatrix_imm(X, y_kkm, gamma, rbf, check_all_cuts = True)

        rand_kmat_imm_on_kkm = adjusted_rand_score(y_kmat_imm_on_kkm, y_true)
        price_kmat_imm_on_kkm = np.sum(kernelkmeanscost(Kmat, y_kmat_imm_on_kkm))/np.sum(kernelkmeanscost(Kmat, y_kkm))
            
        results = {'rand_kmeans': rand_kmeans,
                   'rand_imm': rand_imm,
                   'best_kernel': best_kernel,
                   'best_gamma': gamma,
                   'rand_kkm': rand_kkm,
                   'rand_taylor_imm_on_kkm': rand_taylor_imm_on_kkm,
                   'price_taylor_imm_on_kkm': price_taylor_imm_on_kkm,
                   'rand_kmat_imm_on_kkm': rand_kmat_imm_on_kkm,
                   'price_kmat_imm_on_kkm': price_kmat_imm_on_kkm,
                   'threshold_cuts_taylor': threshold_cuts_taylor,
                   'threshold_cuts_kmat': threshold_cuts_kmat
              }
        
        labels = {'y_kmeans': y_kmeans,
                  'y_kmeans_imm': y_kmeans_imm,
                  'y_kkm': y_kkm,
                  'y_taylor_imm_on_kkm': y_taylor_imm_on_kkm,
                  'y_kmat_imm_on_kkm': y_kmat_imm_on_kkm
                 }
        
        return(results, labels)
    
            
        
    elif best_kernel==1:
        
        print('Run Laplace kernel matrix IMM on kernel k-means')
        y_kmat_imm_on_kkm, threshold_cuts_kmat = kernelmatrix_imm(X, y_kkm, gamma, laplace, check_all_cuts = True)

        rand_kmat_imm_on_kkm = adjusted_rand_score(y_kmat_imm_on_kkm, y_true)
        price_kmat_imm_on_kkm = np.sum(kernelkmeanscost(Kmat, y_kmat_imm_on_kkm))/np.sum(kernelkmeanscost(Kmat, y_kkm))
            
        results = {'rand_kmeans': rand_kmeans,
                   'rand_imm': rand_imm,
                   'best_kernel': best_kernel,
                   'best_gamma': gamma,
                   'rand_kkm': rand_kkm,
                   'rand_kmat_imm_on_kkm': rand_kmat_imm_on_kkm,
                   'price_kmat_imm_on_kkm': price_kmat_imm_on_kkm,
                   'threshold_cuts_kmat': threshold_cuts_kmat
              }
        
        labels = {'y_kmeans': y_kmeans,
                  'y_kmeans_imm': y_kmeans_imm,
                  'y_kkm': y_kkm,
                  'y_kmat_imm_on_kkm': y_kmat_imm_on_kkm
         }
        
        return(results, labels)
        
def refine_imm(X, y_true, y_kkm, y_imm, Kmat, max_leaves):
    
    print('Kernel ExKMC')
    y_greedy = exkmc_build_on_imm(X, y_kkm, y_imm, Kmat, max_leaves)
    y_exkmc = np.zeros(X.shape[0])
    
    for cluster in np.unique(y_greedy):
        index_u = np.where(y_greedy==cluster)[0]
        best_label = exkmc_min_cost_at_node(index_u, Kmat, y_kkm)[1]
        y_exkmc[index_u] = best_label
        
    rand_exkmc = adjusted_rand_score(y_exkmc, y_true)
    price_exkmc = np.sum(kernelkmeanscost(Kmat, y_exkmc))/np.sum(kernelkmeanscost(Kmat, y_kkm))
   
    print('Kernel Expand')
    y_greedy2 = expand_build_on_imm(X, y_kkm, y_imm, max_leaves)
    y_expand = np.zeros(X.shape[0])
    
    for cluster in np.unique(y_greedy2):
        index_u = np.where(y_greedy2==cluster)[0]
        best_label = expand_min_cost_at_node(index_u, y_kkm)[1]
        y_expand[index_u] = best_label

    rand_expand = adjusted_rand_score(y_expand, y_true)
    price_expand = np.sum(kernelkmeanscost(Kmat, y_expand))/np.sum(kernelkmeanscost(Kmat, y_kkm))
    
    results = {'rand_exkmc': rand_exkmc,
               'price_exkmc': price_exkmc,
               'rand_expand': rand_expand,
               'price_expand': price_expand
              }
    
    labels = {'y_exkmc': y_exkmc,
              'y_expand': y_expand
             }
    
    return(results, labels)
    