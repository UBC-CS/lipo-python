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
        
def pure_random_search(func, bounds, n):
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
    
    y = []
    x = []
    best = []
    
    bound_mins = np.array([bnd[0] for bnd in bounds])
    bound_maxs = np.array([bnd[1] for bnd in bounds])
    
    for t in np.arange(n):
        u = np.random.uniform(size=len(bounds))
        x_prop = u * (bound_maxs - bound_mins) + bound_mins
        x.append(x_prop)
        y.append(func(x_prop))
        best.append(np.max(y))
        
    output = {
        'loss': np.array(best).reshape(n),
        'x': np.array(x),
        'y': np.array(y)
    }
    return output

def adaptive_lipo(func, bounds, n, k_seq=np.logspace(-10,10,10000), p=0.5):
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
    k = 0

    #lower_bound = lambda x_prop, y, x, k: np.max(y-k*np.linalg.norm(x_prop-x))
    upper_bound = lambda x_prop, y, x, k: np.min(y+k*np.linalg.norm(x_prop-x))

    for t in np.arange(n):

        # draw a uniformly distributed random variable
        u = np.random.uniform(size=len(bounds))
        x_prop = u * (bound_maxs - bound_mins) + bound_mins

        # check if we are exploring or exploiting
        if not np.random.binomial(n=1, p=p):
            # exploiting - must ensure we're drawing from potential minimizers
            while upper_bound(x_prop, y, x, k) < np.max(y):
                u = np.random.uniform(size=len(bounds))
                x_prop = u * (bound_maxs - bound_mins) + bound_mins
        
        # once settled on proposal add it to the seen points
        x.append(x_prop)
        y.append(func(x_prop))
        best.append(np.max(y))

        # update estimate of lipschitx constant
        # compute pairwise differences between y values
        y_outer = np.outer(np.ones(len(y)), y)
        y_diff = np.abs(y_outer - y_outer.T)
        # compute distance matrix between x values
        x_dist = squareform(pdist(np.array(x)))
        np.fill_diagonal(x_dist, np.Inf)
        # estimate lipschitz constant
        k_est = np.max(y_diff / x_dist)
        k = k_seq[np.argmax(k_seq > k_est)]


    output = {
        'loss': np.array(best).reshape(n),
        'x': np.array(x),
        'y': np.array(y)
    }
    return output
