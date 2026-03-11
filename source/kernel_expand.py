# https://github.com/maxf14/explaining_kernel_clustering/blob/main/ExpandingIMM.ipynb
import numpy as np

def expand_min_cost_at_node(index_u, y, silent = True):
    
    ### minimal error of a given set of points index_u
    
    # y is the array of reference cluster assignments
    
    if len(index_u)==0:
        error_best = 0
        cluster_best = 0
    
    else:
        
        counts = np.zeros(len(np.unique(y))) # counts occurences of each cluster label in index_u
        
        for i in np.arange(len(counts)):
            index_i = np.where(y==i)[0]
            counts[i] = np.sum(np.isin(index_u, index_i))
        
        cluster_best = np.argmax(counts)
        error_best = len(index_u) - np.max(counts)

        if silent == False:
            print(counts)
            print(cluster_best)
            print(error_best)

    return(error_best, cluster_best)

#points = np.where((y_true==1) | (y_true==2))[0]
#points = np.random.choice(len(y_true), 20)
#print(points)
#expand_min_cost_at_node(points, y_true, silent = False)

def expand_cost_delta_of_split(i, theta1, theta2, index_u, X, y, silent = True):
    
    # Given a split (i,theta) we determine the cost of it, over all partitions
    
    # i: Axis we use to split
    # theta1 and theta2: Thresholds we use to split
    # index_u: Index of points at this node
    # X: Data
    # y: Cluster Labels
    # Kmat: Kernel Matrix
    
    cost_old =  expand_min_cost_at_node(index_u, y, silent)[0]
    
    # Row IDs of the interval
    index_L = index_u[np.where((X[np.ix_(index_u,[i])]>=theta1) & (X[np.ix_(index_u,[i])]<=theta2))[0]]
    
    # Row IDs of not the interval 
    index_R = index_u[np.where((X[np.ix_(index_u,[i])]<theta1) | (X[np.ix_(index_u,[i])]>theta2))[0]]
    
    cost_new = expand_min_cost_at_node(index_L, y, silent)[0] + expand_min_cost_at_node(index_R, y, silent)[0]
    
    cost_delta = cost_old - cost_new # the larger the better
    
    if silent == False:
        print('Cost old:', cost_old)
        print('Cost new:', cost_new)
        print('Cost delta:', cost_delta)
    
    return(cost_delta)

#print(expand_cost_delta_of_split(0, 0, 10, np.arange(X.shape[0]), X, y_true, silent = False)) # bad
#print(expand_cost_delta_of_split(0, 0, 1.5, np.arange(X.shape[0]), X, y_true, silent = False)) # good

def expand_split_node(index_u, X, y, silent = True):
    
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
                delta_new = expand_cost_delta_of_split(i, theta1, theta2, index_u, X, y)
            
                if delta_new > best_delta:

                    if silent == False:
                        print('Improvement at theta1 =', theta1, 'and theta2=', theta2)
                    
                    best_delta = delta_new
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
#print(points)
#plt.scatter(X[points, 0], X[points, 1], c=y_true[points], s=50, cmap='viridis')
#expand_split_node(points, X, y_true, silent = False)

def expand_new_cut(X, y, index_nodes, info_nodes, silent=True):
    
    ### This function adds a new cut no matter what
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
                
            print('---> Best delta = ', best_delta_u)

            if best_delta_u > best_delta:

                best_delta = best_delta_u
                best_u = u
                best_cut_i = best_i_u
                best_cut_theta1 = best_theta1_u
                best_cut_theta2 = best_theta2_u
    
    print('--- NEW CUT --- ... at node', best_u, 
          '... at Coordinate', best_cut_i, 
          '... at Thresholds', best_cut_theta1, best_cut_theta2,
          '... for delta =', best_delta)

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
    
    info_update = info_nodes.copy()
    info_update[best_u] = 'Empty' 
    info_update.append('Empty')

    return(index_update, info_update)

#points = np.where((y_true==1) | (y_true==2))[0]
#expand_new_cut(X, y_true, [points], silent = True)

def expand_build_on_imm(X, y, y_imm, max_leaves, silent = True):
    
    n_data = np.shape(X)[0]
    index_nodes = []
    info_nodes = []
    
    for i in np.unique(y_imm):
        
        indices_i = np.where(y_imm == i)[0]
        index_nodes.append(indices_i)
        info_nodes.append('Empty')
    
    # index_nodes is a partition that contains the IMM clusters
    # we will further partition it using 'expand_new_cut'
    
    converged = False
    
    if(len(index_nodes)>=max_leaves):
        converged = True
        print('Converged')
    
    while converged == False:
        
        index_nodes, info_nodes = expand_new_cut(X, y, index_nodes, info_nodes)
        
        if(len(index_nodes)>=max_leaves):
            converged = True
            print('Converged')

    y_greedy = np.zeros(n_data)
    
    for i in np.arange(len(index_nodes)):
        index_u = index_nodes[i]
        y_greedy[index_u] = i
    
    return(y_greedy)
