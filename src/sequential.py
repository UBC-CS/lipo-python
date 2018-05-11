"""
Sequential algorithms for minimizing expensive functions
"""

import numpy as np

def lipo(func, bounds, k, n):
    """
    Parameters
    ----------
     - func:   the (expensive) function to be minimized
     - bounds: list of tuples containing boundaries defining the domain of f 
     - k:      the lipschitz constant of f
     - n:      number of iterations to perform
    Returns
    ------
     - x within bounds that returned smallest value f(x)
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

    lower_bound = lambda x_prop, y, x, k: np.max(y-k*np.linalg.norm(x_prop-x))
    
    # iteration
    for t in np.arange(n):
        u = np.random.uniform(size=len(bounds))
        x_prop = u * (bound_maxs - bound_mins) + bound_mins
        if lower_bound(x_prop, y, x, k) <= np.min(y):
            x.append(x_prop)
            y.append(func(x_prop))
        best.append(np.min(y))

    output = {
        'loss': np.array(best).reshape(n),
        'x': np.array(x),
        'y': np.array(y)
    }
    return output
        
def prs(func, bounds, n):
    """
    Pure Random Search
    
    Parameters
    ----------
     - func:   the (expensive) function to be minimized
     - bounds: list of tuples containing boundaries defining the domain of f 
     - n:      number of iterations to perform
    Returns
    ------
     - x within bounds that returned smallest value f(x)
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
        best.append(np.min(y))
        
    output = {
        'loss': np.array(best).reshape(n),
        'x': np.array(x),
        'y': np.array(y)
    }
    return output

def adaptive_lipo(func, bounds, n, k_seq, p):
    """
    Parameters
    ----------
     - func:   the (expensive) function to be minimized
     - bounds: list of tuples containing boundaries defining the domain of f 
     - k_seq:  nondecreasing sequence of lipschitz constants
     - n:      number of iterations to perform
     - p:      parameter of the binomial dist. controlling exploration/exploitation
    Returns
    ------
     - x within bounds that returned smallest value f(x)
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

    lower_bound = lambda x_prop, y, x, k: np.max(y-k*np.linalg.norm(x_prop-x))

    for t in np.arange(t):
        b = np.random.binomial(n=1, p=p)
        if b == 1:
            u = np.random.uniform(size=len(bounds))
            x_prop = u * (bound_maxs - bound_mins) + bound_mins
        else:
            # randomly sample from the set of potential minimizers
            # element x contained within bounds such that there exists 
            # a lipschitz continuous function g that equals our y = f(x_t) 
            # values for all t and the element x minimizes the function g
            # how (if at all) can this be linked to the lower bound?
        
        x.append(x_prop)
        y.append(func(x_prop))
        best.append(np.min(y))

        # update estimate of the lipschitz constant
        # should be the minimum value in k_seq 
        # that is compatible with our observed x, y

    output = {
        'loss': np.array(best).reshape(n),
        'x': np.array(x),
        'y': np.array(y)
    }
    return output
