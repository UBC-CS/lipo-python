"""
Objective functions and domains for comparing the sequential algorithms
"""

import numpy as np

def holder_table(x):
    """x is a numpy array containing two numbers"""
    if x.shape != (2,):
        raise ValueError('Input array should be shape of (2,)')
    inside_exp = np.abs(1-np.sqrt(x[0]*x[0]+x[1]*x[1])/np.pi)
    return np.abs(np.sin(x[0])*np.cos(x[1])*np.exp(inside_exp))

holder_bounds = [(-10,10),(-10,10)]

def rosenbrock(x):
    """x is a numpy array containing three numbers"""
    if x.shape != (3,):
        raise ValueError('Input array should be shape of (3,)')
    first = 100*(x[1] - x[0]**2)**2 + (x[0] - 1)**2
    second = 100*(x[2] - x[1]**2)**2 + (x[1] - 1)**2
    return -np.sum(first + second)

rosenbrock_bounds = [(-2.048,2.048), (-2.048,2.048), (-2.048,2.048)]

def sphere(x):
    """x is a numpy array containing four numbers"""
    if x.shape != (4,):
        raise ValueError('Input array should be shape of (4,)')
    return -np.sqrt(np.sum((x - np.pi/16)**2))

sphere_bounds = [(0,1), (0,1), (0,1), (0,1)]

def linear_slope(x):
    """x is a numpy array containing four numbers"""
    if x.shape != (4,):
        raise ValueError('Input array should be shape of (4,)')
    coef = np.array([10**((i - 1)/4) for i in np.arange(4)])
    return np.sum(coef*x)

linear_slope_bounds = [(-5,5), (-5,5), (-5,5), (-5,5)]

def deb_one(x):
    """x is a numpy array containing five numbers"""
    if x.shape != (5,):
        raise ValueError('Input array should be shape of (5,)')
    return (1/5)*np.sum(np.sin(5*np.pi*x)**6) 

deb_one_bounds = [(-5,5), (-5,5), (-5,5), (-5,5), (-5,5)]

synthetic_functions = {
    'Holder Table' : {'func': holder_table, 'bnds': holder_bounds},
    'Rosenbrock': {'func': rosenbrock, 'bnds': rosenbrock_bounds},
    'Sphere': {'func': sphere, 'bnds': sphere_bounds},
    'Linear Slope': {'func': linear_slope, 'bnds': linear_slope_bounds},
    'Deb N.1': {'func': deb_one, 'bnds': deb_one_bounds}
}
