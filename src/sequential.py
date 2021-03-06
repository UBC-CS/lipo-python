"""
Sequential algorithms for maximizing expensive functions
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
        
def pure_random_search(func, bounds, n, seed=None):
    """
    Pure Random Search
    
    Parameters
    ----------
     - func:   the (expensive) function to be maximized
     - bounds: list of tuples containing boundaries defining the domain of f 
     - n:      number of iterations to perform
    Returns
    ------
     - x within bounds that returned largest value f(x)
    """
    np.random.seed(seed)

    # dimension of the domain
    d = len(bounds)
    
    # preallocate the output arrays
    y = np.zeros(n) - np.Inf
    x = np.zeros((n, d))
    loss = np.zeros(n)

    # the lower/upper bounds on each dimension
    bound_mins = np.array([bnd[0] for bnd in bounds])
    bound_maxs = np.array([bnd[1] for bnd in bounds])
    
    for t in np.arange(n):
        u = np.random.uniform(size=d)
        x_prop = u * (bound_maxs - bound_mins) + bound_mins
        x[t] = x_prop
        y[t] = func(x_prop)
        loss[t] = np.max(y)
        
    output = {'loss': loss, 'x': x, 'y': y}
    return output

def adaptive_lipo(func, 
                  bounds, 
                  n, 
                  p=0.1,
                  seed=None):
    """
    Parameters
    ----------
     - func:   the (expensive) function to be maximized
     - bounds: list of tuples containing boundaries defining the domain of f 
     - k_seq:  nondecreasing sequence of lipschitz constants
     - n:      number of iterations to perform
     - p:      parameter of the binomial dist. controlling exploration/exploitation
    Returns
    ------
     - x within bounds that returned largest value f(x)
    """
    np.random.seed(seed)

    # dimension of the domain
    d = len(bounds)

    alpha = 0.01/d
    k_seq=(1+alpha)**np.arange(-10000,10000) # Page 16

    # preallocate the output arrays
    y = np.zeros(n) - np.Inf
    x = np.zeros((n, d))
    loss = np.zeros(n)
    k_arr = np.zeros(n)

    # preallocate the distance arrays
    # x_dist = np.zeros((n * (n - 1)) // 2)
    # y_dist = np.zeros((n * (n - 1)) // 2)
    
    # the lower/upper bounds on each dimension
    bound_mins = np.array([bnd[0] for bnd in bounds])
    bound_maxs = np.array([bnd[1] for bnd in bounds])
    
    # initialization with randomly drawn point in domain and k = 0
    k = 0
    k_est = -np.inf
    u = np.random.rand(d)
    x_prop = u * (bound_maxs - bound_mins) + bound_mins
    x[0] = x_prop
    y[0] = func(x_prop)
    k_arr[0] = k

    upper_bound = lambda x_prop, y, x, k: np.min(y+k*np.linalg.norm(x_prop-x,axis=1))

    for t in np.arange(1, n):
        print('Iteration {}'.format(t))

        # draw a uniformly distributed random variable
        u = np.random.rand(d)
        x_prop = u * (bound_maxs - bound_mins) + bound_mins

        # check if we are exploring or exploiting
        if np.random.rand() > p: # enter to exploit w/ prob (1-p)
            # exploiting - ensure we're drawing from potential maximizers
            print('Into the while...')
            while upper_bound(x_prop, y[:t], x[:t], k) < np.max(y[:t]):
                u = np.random.rand(d)
                x_prop = u * (bound_maxs - bound_mins) + bound_mins 
                # import pdb
                # pdb.set_trace()

                # print(upper_bound(x_prop, y[:t], x[:t], k))
            #     print(np.max(y[:t]))
                # print('------')
            print('Out of the while')
        else:
            pass 
            # we keep the randomly drawn point as our next iterate
            # this is "exploration"
        # add proposal to array of visited points
        x[t] = x_prop
        y[t] = func(x_prop)
        loss[t] = np.max(y)

        # compute current number of tracked distances
        old_num_dist = (t * (t - 1)) // 2
        new_num_dist = old_num_dist + t

        # compute distance between newl values and all 
        # previously seen points - should be of shape (t,)
        # then insert new distances into x_dist, y_dist resp.
        # new_x_dist = np.linalg.norm(x[:t]-x[t],axis=1)
        new_x_dist = np.sqrt(np.sum((x[:t] - x[t])**2, axis=1))
        # x_dist[old_num_dist:new_num_dist] = new_x_dist 

        new_y_dist = np.abs(y[:t] - y[t])
        # y_dist[old_num_dist:new_num_dist] = new_y_dist

        # compute new number of of tracked distances
        # and update estimate of lipschitz constant
        k_est = max(k_est, np.max(new_y_dist/new_x_dist))  # faster
        # k_est = np.max(y_dist[:new_num_dist] / x_dist[:new_num_dist])  # slow because num_dist gets large
        

        # get the smallest element of k_seq that is bigger than k_est
        # note: we're using argmax to get the first occurrence
        # note: this relies on k_seq being sorted in nondecreasing order
        k = k_seq[np.argmax(k_seq >= k_est)]
        # note: in the paper, k_seq is called k_i 
        #       and k is called \hat{k}_t
        # print(k)
        i_t = np.ceil(np.log(k_est)/np.log(1+alpha))
        k = (1+alpha)**i_t
        print('Lipschitz Constant Estimate: {}'.format(k))
        print('\n')
        # print(k)
        # print("----")
        k_arr[t] = k
        

    output = {'loss': loss, 'x': x, 'y': y, 'k': k_arr}
    return output

optimizers = {
    'AdaLIPO': adaptive_lipo,
    'PRS': pure_random_search
}