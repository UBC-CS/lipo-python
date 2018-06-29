"""
Objective functions and domains for comparing the sequential algorithms
"""

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_validate

def holder_table(x):
    if x.shape[0] != 2:
        raise ValueError('Input array first dimension should be size 2')
    inside_exp = np.abs(1-np.sqrt(x[0]*x[0]+x[1]*x[1])/np.pi)
    return np.abs(np.sin(x[0])*np.cos(x[1])*np.exp(inside_exp))

holder_bounds = [(-10,10),(-10,10)]

def rosenbrock(x):
    if x.shape[0] != 3:
        raise ValueError('Input array first dimension should be size 3')
    first = 100*(x[1] - x[0]**2)**2 + (x[0] - 1)**2
    second = 100*(x[2] - x[1]**2)**2 + (x[1] - 1)**2
    return -(first + second)

rosenbrock_bounds = [(-2.048,2.048), (-2.048,2.048), (-2.048,2.048)]

def sphere(x):
    if x.shape[0] != 4:
        raise ValueError('Input array first dimension should be size 4')
    return -np.sqrt(np.sum((x - np.pi/16)**2, axis=0))

sphere_bounds = [(0,1), (0,1), (0,1), (0,1)]

def linear_slope(x):
    if x.shape[0] != 4:
        raise ValueError('Input array first dimension should be size 4)')

    coef = 10.0**(np.arange(4)/4)
    return coef@(x-5)

linear_slope_bounds = [(-5,5), (-5,5), (-5,5), (-5,5)]

def deb_one(x):
    if x.shape[0] != 5:
        raise ValueError('Input array first dimension should be of size 5')
    return (1/5)*np.sum(np.sin(5*np.pi*x)**6, axis=0)

deb_one_bounds = [(-5,5), (-5,5), (-5,5), (-5,5), (-5,5)]

synthetic_functions = {
    'Holder Table' : {'func': holder_table, 'bnds': holder_bounds},
    'Rosenbrock': {'func': rosenbrock, 'bnds': rosenbrock_bounds},
    'Linear Slope': {'func': linear_slope, 'bnds': linear_slope_bounds},
    'Sphere': {'func': sphere, 'bnds': sphere_bounds},
    'Deb N.1': {'func': deb_one, 'bnds': deb_one_bounds}
}

def get_data(path):
    full = pd.read_csv(path)
    y = full.ix[:, 0]
    X = full.ix[:, range(1, len(full.columns))]
    return X, y

def kernel_ridge_CV(X, y, cv, params):
    """Kernel Ridge regression on an arbitrary dataset as a function
    of the gaussian kernel bandwidth and regularization strength"""
    # switch from math land to code land
    alpha = params[0]
    gamma = 1/(2*params[1]**2)
    # build model and do cross validation
    model = KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma)
    results = cross_validate(model, X, y, cv=cv)
    return np.mean(results['test_score'])

X_housing, y_housing = get_data('./data/clean/housing.csv')

def housing(x):
    return kernel_ridge_CV(X_housing, y_housing, 10, x)

X_yacht, y_yacht = get_data('./data/clean/yacht_hydrodynamics.csv')

def yacht(x):
    return kernel_ridge_CV(X_yacht, y_yacht, 10, x)

real_bnds = [(-2, 4), (-5, 5)]

objectives = {
    'Holder Table': {'func': holder_table, 'bnds': holder_bounds},
    'Rosenbrock': {'func': rosenbrock, 'bnds': rosenbrock_bounds},
    'Linear Slope': {'func': linear_slope, 'bnds': linear_slope_bounds},
    'Sphere': {'func': sphere, 'bnds': sphere_bounds},
    'Deb N.1': {'func': deb_one, 'bnds': deb_one_bounds},
    'Housing': {'func': housing, 'bnds': real_bnds},
    'Yacht': {'func': housing, 'bnds': real_bnds}
}
