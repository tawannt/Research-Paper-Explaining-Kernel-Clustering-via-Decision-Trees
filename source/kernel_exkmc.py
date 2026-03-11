# https://github.com/maxf14/explaining_kernel_clustering/blob/main/KernelExKMC.ipynb
import numpy as np
from kernel_expand import expand_split_node

def cost_points2cluster(index_u, cluster, Kmat, y):
    
    ### cost of assigning points to a given cluster
    cluster_index = np.where(y == cluster)[0] # get the points in cluster
    #print(cluster_index)
    
    normcluster2 = np.sum(Kmat[np.ix_(cluster_index, cluster_index)])/len(cluster_index)**2
    
    index_dot_cluster = np.sum(Kmat[np.ix_(index_u, cluster_index)])/len(cluster_index)
    #print(Kmat[np.ix_(index_u, cluster_index)].shape)
    
    sum_xx = np.sum(np.diagonal(Kmat[np.ix_(index_u, index_u)]))
    
    return(sum_xx + normcluster2 - 2*index_dot_cluster)

#points = np.where(y_true==1)[0]
#cost_points2cluster(points, 2, Kmat, y_true)

def exkmc_min_cost_at_node(index_u, Kmat, y, silent = True):
    
    ### minimal cost of a set of points
    
    cost_best = float('inf')
    
    for cluster in np.unique(y):
        
        cost_u = cost_points2cluster(index_u, cluster, Kmat, y)
    
        if cost_u < cost_best:
            cluster_best = cluster
            cost_best = cost_u
    
    return(cost_best, cluster_best)

#points = np.where(y_true==1)[0]
#exkmc_min_cost_at_node(points, Kmat, y_true, silent = False)[0]

def exkmc_cost_delta_of_split(i, theta1, theta2, index_u, X, y, Kmat, silent = True):
    
    # Given a split (i,theta) we determine the cost of it, over all partitions
    
    # i: Axis we use to split
    # theta1 and theta2: Thresholds we use to split
    # index_u: Index of points at this node
    # X: Data
    # y: Cluster Labels
    # Kmat: Kernel Matrix
    
    # Current cost of a given node
    cost_u = exkmc_min_cost_at_node(index_u, Kmat, y)[0]
    
    #print('Current cost at this node:', cost_u)
    
    # Row IDs of the interval
    index_L = index_u[np.where((X[np.ix_(index_u,[i])]>=theta1) & (X[np.ix_(index_u,[i])]<=theta2))[0]]
    
    # Row IDs of not the interval 
    index_R = index_u[np.where((X[np.ix_(index_u,[i])]<theta1) | (X[np.ix_(index_u,[i])]>theta2))[0]]
    
    cost_new = exkmc_min_cost_at_node(index_L, Kmat, y)[0] + exkmc_min_cost_at_node(index_R, Kmat, y)[0]
    
    cost_delta = cost_u - cost_new # we will choose the largest cost_delta (i.e. lowest cost_new)
    
    if silent == False:
        print('Old cost:', cost_u)
        print('New cost:', cost_new)
        print('Delta:', cost_u - cost_new)
    
    return(cost_delta)

#exkmc_cost_delta_of_split(0, 0, 10, np.arange(X.shape[0]), X, y_true, Kmat, silent = False) # bad
#exkmc_cost_delta_of_split(0, 0, 1, np.arange(X.shape[0]), X, y_true, Kmat, silent = False) # good

def exkmc_split_node(index_u, X, y, Kmat, silent = True):
    
    # Splits a given node at the best possible threshold cut
    
    # index_u: points at this node
    
    best_delta = (-1)*float('inf')
    
    for i in np.arange(np.shape(X)[1]):
        
        print('Check Coordinate',i)
        thetas = np.unique(np.sort(X[np.ix_(index_u,[i])]))
        
        #print('Thresholds:', thetas)
        
        for theta1 in thetas[:-1]:
            for theta2 in thetas[thetas>theta1]:
                
                #print(theta1, theta2)
                cost_delta = exkmc_cost_delta_of_split(i, theta1, theta2, index_u, X, y, Kmat)
            
                if cost_delta > best_delta:

                    if silent == False:
                        print('Improvement at theta1 =', theta1, 'and theta2=', theta2)
                    
                    best_delta = cost_delta
                    best_i = i
                    best_theta1 = theta1
                    best_theta2 = theta2
                    
    info_u = {'best_delta': best_delta,
              'best_i': best_i,
              'best_theta1': best_theta1,
              'best_theta2': best_theta2
             }
    
    return(info_u)

#points = np.arange(X.shape[0])
#points = np.where((y_true==1) | (y_true==2))[0]
#plt.scatter(X[points, 0], X[points, 1], c=y_true[points], s=50, cmap='viridis')
#exkmc_split_node(points, X, y_true, Kmat, silent = False)

def exkmc_new_cut(X, y, Kmat, index_nodes, info_nodes, silent=True):
    
    ### This function uses ExKMC to add a new cut no matter what
    ### Make sure to only call it as long as you want more leaves
    
    ### X,y are simply the data and the labels
    ### index_nodes is a list of arrays, each containing the index of points at a node
    
    print('+++ Currently, we have', len(index_nodes), 'node(s). Let us find a new cut! +++')
    
    best_delta = (-1)*float('inf')
    n_nodes = len(index_nodes)
    
    for u in np.arange(n_nodes):
        
        print('--- Check node', u, '---')
        
        index_u = index_nodes[u]
        print('Number of points at this node:', len(index_u))
        
        info_u = info_nodes[u]
        print('Available info at this node:', info_u)
        
        if len(index_u)<=3:
            best_delta_u = (-1)*float('inf') # don't further cut small leafs
            
        else:
            
            if info_u == 'Empty':
                print('Fetch info on this new node')
                info_u = expand_split_node(index_u, X, y)
                info_nodes[u] = info_u # write this info to the big list
            
            # anyways this is what we are interested in!
            best_delta_u = info_u['best_delta']
            best_i_u = info_u['best_i']
            best_theta1_u = info_u['best_theta1']
            best_theta2_u = info_u['best_theta2']
        
            print('---> Delta = ', best_delta_u)

            if best_delta_u > best_delta:

                best_delta = best_delta_u
                best_u = u
                best_cut_i = best_i_u
                best_cut_theta1 = best_theta1_u
                best_cut_theta2 = best_theta2_u
    
    print('--- NEW CUT --- ... at node', best_u, 
          '... at Coordinate', best_cut_i, 
          '... at Thresholds', best_cut_theta1, best_cut_theta2)

    #go_left_u = np.where(X[np.ix_(index_nodes[best_u],[best_cut_i])]<=best_cut_theta)[0]
    go_left_u = np.where((X[np.ix_(index_nodes[best_u],[best_cut_i])]>=best_cut_theta1) & 
                         (X[np.ix_(index_nodes[best_u],[best_cut_i])]<=best_cut_theta2))[0]
    #print('Send Left:', (index_nodes[best_u])[go_left_u])
    
    #go_right_u = np.where(X[np.ix_(index_nodes[best_u],[best_cut_i])]>best_cut_theta)[0]
    go_right_u = np.where((X[np.ix_(index_nodes[best_u],[best_cut_i])]<best_cut_theta1) |
                          (X[np.ix_(index_nodes[best_u],[best_cut_i])]>best_cut_theta2))[0]
    #print('Send Right:', (index_nodes[best_u])[go_right_u])

    index_update = index_nodes.copy()
    index_update[best_u] = (index_nodes[best_u])[go_left_u]
    index_update.append((index_nodes[best_u])[go_right_u])
    
    print('Node left:', len(go_left_u))
    print('Node right:', len(go_right_u))
    
    info_update = info_nodes.copy()
    info_update[best_u] = 'Empty' 
    info_update.append('Empty')

    return(index_update, info_update)

#points = np.where((y_true==1) | (y_true==2))[0]
#exkmc_new_cut(X, y_true, Kmat, [points], silent = True)

def exkmc_build_on_imm(X, y, y_imm, Kmat, max_leaves, silent = True):
    
    n_data = np.shape(X)[0]
    index_nodes = []
    info_nodes = []
    
    for i in np.unique(y_imm):
        
        indices_i = np.where(y_imm == i)[0]
        index_nodes.append(indices_i)
        info_nodes.append('Empty')
    
    # index_nodes is a partition that contains the IMM clusters
    # we will further partition it using 'exkmc_new_cut'
    
    converged = False
    
    if(len(index_nodes)>=max_leaves):
        converged = True
        print('Converged')
    
    while converged == False:
        
        index_nodes, info_nodes = exkmc_new_cut(X, y, Kmat, index_nodes, info_nodes)
        
        if(len(index_nodes)>=max_leaves):
            converged = True
            print('Converged')

    y_greedy = np.zeros(n_data)
    
    for i in np.arange(len(index_nodes)):
        index_u = index_nodes[i]
        y_greedy[index_u] = i
    
    return(y_greedy)
