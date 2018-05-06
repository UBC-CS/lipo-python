#!usr/bin/env python

import numpy as np

def lower_bound(x_prop, y, x, k):
    """
    Parameters
    ----------
     - x_prop: candidate point at which to evaluate f
     - y:      values of f already seen
     - x:      values in domain correponding to y
     - k:      lipschitz constant of f
    Returns
    -------
     - lower bound on f 
    """
    return np.max(y - k * np.linalg.norm(x_prop - x))

def LIPO(f, bounds, k, n, seq_out=False):
    """
    Parameters
    ----------
     - f:      the (expensive) function to be minimized
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
    y.append(f(x[0]))
    
    # iteration
    for t in np.arange(n):
        u = np.random.uniform(size=len(bounds))
        x_prop = u * (bound_maxs - bound_mins) + bound_mins
        if lower_bound(x_prop, y, x, k) <= np.min(y):
            x.append(x_prop)
            y.append(f(x_prop))
        best.append(x[np.array(y).argmin()])
            
    # output
    if seq_out:
        return np.array(best).reshape(n)
    else:
        return x[np.array(y).argmin()]