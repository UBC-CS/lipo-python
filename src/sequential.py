"""
Sequential algorithms for maximizing expensive functions
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform

def lipo(func, bounds, k, n):
    """
    Parameters
    ----------
     - func:   the (expensive) function to be maximized
     - bounds: list of tuples containing boundaries defining the domain of f 
     - k:      the lipschitz constant of f
     - n:      number of iterations to perform
    Returns
    ------
     - x within bounds that returned largest value f(x)
    """
    
    # initialization
    y = []
    x = []
    best = []
    
    bound_mins = np.array([bnd[0] for bnd in bounds])
    bound_maxs = np.array([bnd[1] for bnd in bounds])
    
    u = np.random.uniform(size=len(bounds))
    x_prop = u * (bound_maxs - bound_mins) + bound_mins
    
    x.append(x_prop)
    y.append(func(x[0]))

    #lower_bound = lambda x_prop, y, x, k: np.max(y-k*np.linalg.norm(x_prop-x))
    upper_bound = lambda x_prop, y, x, k: np.min(y+k*np.linalg.norm(x_prop-x))
    
    # iteration
    for t in np.arange(n):
        u = np.random.uniform(size=len(bounds))
        x_prop = u * (bound_maxs - bound_mins) + bound_mins
        if upper_bound(x_prop, y, x, k) >= np.max(y):
            x.append(x_prop)
            y.append(func(x_prop))
        best.append(np.max(y))

    output = {
        'loss': np.array(best).reshape(n),
        'x': np.array(x),
        'y': np.array(y)
    }
    return output
        
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
                  k_seq=np.array([(1 + (0.01/0.5))**i for i in range(-10000, 10000)]), 
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

    # preallocate the output arrays
    y = np.zeros(n) - np.Inf
    x = np.zeros((n, d))
    loss = np.zeros(n)

    # preallocate the distance arrays
    x_dist = np.zeros((n * (n - 1)) // 2)
    y_dist = np.zeros((n * (n - 1)) // 2)
    
    # the lower/upper bounds on each dimension
    bound_mins = np.array([bnd[0] for bnd in bounds])
    bound_maxs = np.array([bnd[1] for bnd in bounds])
    
    # initialization with randomly drawn point in domain and k = 0
    k = 0
    u = np.random.rand(d)
    x_prop = u * (bound_maxs - bound_mins) + bound_mins
    x[0] = x_prop
    y[0] = func(x_prop)

    upper_bound = lambda x_prop, y, x, k: np.min(y+k*np.linalg.norm(x_prop-x,axis=1))

    for t in np.arange(1, n):

        # draw a uniformly distributed random variable
        u = np.random.rand(d)
        x_prop = u * (bound_maxs - bound_mins) + bound_mins

        # check if we are exploring or exploiting
        if not np.random.binomial(n=1, p=p):
            # exploiting - ensure we're drawing from potential maximizers
            while upper_bound(x_prop, y[:t], x[:t], k) < np.max(y):
                u = np.random.rand(d)
                x_prop = u * (bound_maxs - bound_mins) + bound_mins 

        # add proposal to array of visited points
        x[t] = x_prop
        y[t] = func(x_prop)
        loss[t] = np.max(y)

        # compute current number of tracked distances
        old_num_dist = (t * (t - 1)) // 2

        # compute distance between newl values and all 
        # previously seen points - should be of shape (t,)
        # then insert new distances into x_dist, y_dist resp.
        new_x_dist = np.sqrt(np.sum((x[:t] - x_prop)**2, axis=1))
        x_dist[old_num_dist:(old_num_dist + t)] = new_x_dist 

        new_y_dist = np.abs(y[:t] - y[t])
        y_dist[old_num_dist:(old_num_dist + t)] = new_y_dist

        # compute new number of of tracked distances
        # and update estimate of lipschitz constant
        new_num_dist = old_num_dist + t
        k_est = np.max(y_dist[:new_num_dist] / x_dist[:new_num_dist]) 
        k = k_seq[np.argmax(k_seq > k_est)]


    output = {'loss': loss, 'x': x, 'y': y}
    return output

optimizers = {
    'AdaLIPO': adaptive_lipo,
    'PRS': pure_random_search
}