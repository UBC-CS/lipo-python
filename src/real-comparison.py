#!usr/bin/env python

# Script to replicate the results of the numerical experiments
#
# Usage: python real-comparison.py filename='imagename.png' [--num_sim, --num_iter]

from sequential import optimizers
from objective_functions import kernel_ridge_CV, get_data

import numpy as np
import argparse
import pickle
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('outputfile', type=str,
                    help='name of file to which we write results')
parser.add_argument('--num_sim', type=int, default=20,
                    help='number of runs of sequential optimizers to perform')
parser.add_argument('--num_iter', type=int, default=100,
                    help='number of iterations in each run of a sequential optimizer')
parser.add_argument('--optimizer', type=str,
                    help='sequential optimization algorithm to use',
                    choices=['PRS', 'AdaLIPO'])
args = parser.parse_args()

def recursive_dd():
    return defaultdict(recursive_dd)

X_housing, y_housing = get_data('./data/clean/housing.csv')

def housing(x):
    return kernel_ridge_CV(X_housing, y_housing, 10, x)

X_yacht, y_yacht = get_data('./data/clean/yacht_hydrodynamics.csv')

def yacht(x): 
    return kernel_ridge_CV(X_yacht, y_yacht, 10, x)

real_bnds = [(-2, 4), (-5, 5)]

real_functions = {
    'Housing': {'func': housing, 'bnds': real_bnds},
    'Yacht': {'func': yacht, 'bnds': real_bnds}
}

if __name__ == "__main__":

    if args.num_sim < 1:
        raise RuntimeError('Number of simulations should be a positive integer')
        
    if args.num_iter < 1:
        raise RuntimeError('Number of iterations should be a positive integer')
    
    if args.optimizer:
        seq_optimizers = {args.optimizer: optimizers[args.optimizer]}
    else:
        seq_optimizers = optimizers

    results = recursive_dd()

    # loop over our sequential algortihms
    for optimizer_name, optimizer in seq_optimizers.items():
        # loop over the objective functions
        for real_name, real_obj in real_functions.items():
            # perform specified number of simulations
            for sim in np.arange(args.num_sim):

                out = optimizer(func=real_obj['func'], 
                                bounds=real_obj['bnds'], 
                                n=args.num_iter)
                results[optimizer_name][real_name][sim] = out
    
    # serialize
    # note that if you want to load this serialized object you 
    # need to have recursive_dd defined on the other end
    with open(args.outputfile + '.pkl', 'wb') as place:
        pickle.dump(results, place, protocol=pickle.HIGHEST_PROTOCOL)
             