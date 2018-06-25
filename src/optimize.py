#!usr/bin/env python

# Script to replicate the results of the numerical experiments
#
# Usage: 
#    python optimize.py outputfile 
#    optional args: [--num_sim, --num_iter, --optimizer, --objectives, --synthetic]

from sequential import optimizers
from objective_functions import objectives

import numpy as np
import argparse
import pickle
from tqdm import tqdm

def main(num_sim, num_iter, optimizers, objectives):

    results = dict()
    for func_name, objective in tqdm(objectives.items(),
                                     desc='Objectives', ncols=75):
        
        results[func_name] = dict()
        for optimizer_name, optimizer in tqdm(optimizers.items(),
                                              desc='Optimizers', ncols=75):
            
            results[func_name][optimizer_name] = dict()
            for sim in tqdm(np.arange(num_sim), desc='Simulation', ncols=75):

                out = optimizer(func=objective['func'], 
                                bounds=objective['bnds'], 
                                n=num_iter)
                results[func_name][optimizer_name][sim] = out
    
    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('outputfile', type=str,
                        help='name of file to which we write results')
    parser.add_argument('--num_sim', type=int, default=100,
                        help='number of runs of sequential optimizers to perform')
    parser.add_argument('--num_iter', type=int, default=1000,
                        help='number of iterations in each run of a sequential optimizer')
    parser.add_argument('--optimizer', type=str,
                        help='sequential optimization algorithm to use',
                        choices=['PRS', 'AdaLIPO'])
    parser.add_argument('--objective', type=str, 
                        help='type of objective functions to optimize',
                        choices=['Holder Table', 'Rosenbrock', 'Sphere', 
                                 'Linear Slope', 'Deb N.1', 'Housing', 'Yacht'])
    parser.add_argument('--synthetic', default=False, action='store_true')
    args = parser.parse_args()

    if args.num_sim < 1:
        raise RuntimeError('Number of simulations should be a positive integer')
        
    if args.num_iter < 1:
        raise RuntimeError('Number of iterations should be a positive integer')
    
    if args.optimizer:
        seq_optimizers = {args.optimizer: optimizers[args.optimizer]}
    else:
        seq_optimizers = optimizers
    
    if args.objective:
        objective_funcs = {args.objective: objectives[args.objective]}
    else:
        objective_funcs = objectives

    if args.synthetic:
        objective_funcs = dict()
        funcs = ['Holder Table', 'Rosenbrock', 'Sphere', 'Linear Slope', 'Deb N.1']
        for func in funcs:
            objective_funcs[func] = objectives[func]

    results = main(args.num_sim, args.num_iter, seq_optimizers, objective_funcs)
    print('\n')

    with open(args.outputfile + '.pkl', 'wb') as place:
        pickle.dump(results, place, protocol=pickle.HIGHEST_PROTOCOL)