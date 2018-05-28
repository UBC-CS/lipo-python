"""
Objective functions and domains for comparing the sequential algorithms
"""

import numpy as np

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
    coef = np.array([10**((i - 1)/4) for i in np.arange(4)])
    if len(x.shape) == 1:
        out = np.sum(coef*x)
    else:
        out = np.sum(coef[:,None]*x, axis=0)
    return out

linear_slope_bounds = [(-5,5), (-5,5), (-5,5), (-5,5)]

def deb_one(x):
    if x.shape[0] != 5:
        raise ValueError('Input array first dimension should be of size 5')
    return (1/5)*np.sum(np.sin(5*np.pi*x)**6, axis=0) 

deb_one_bounds = [(-5,5), (-5,5), (-5,5), (-5,5), (-5,5)]

synthetic_functions = {
    'Holder Table' : {'func': holder_table, 'bnds': holder_bounds},
    'Rosenbrock': {'func': rosenbrock, 'bnds': rosenbrock_bounds},
    'Sphere': {'func': sphere, 'bnds': sphere_bounds},
    'Linear Slope': {'func': linear_slope, 'bnds': linear_slope_bounds},
    'Deb N.1': {'func': deb_one, 'bnds': deb_one_bounds}
}
