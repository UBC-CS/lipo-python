#!/usr/bin/env python

from sequential import optimizers
from objective_functions import synthetic_functions

import numpy as np
import argparse
import pickle
import pandas as pd
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('inputfile', type=str,
                    help='input file (results from running synthetic-comparison.py)')
parser.add_argument('outputfile', type=str,
                    help='outputfile')
parser.add_argument('--target', type=float,
                    help='''float between 0 and 1 indicating target value we seek to 
                            reach in optimization process''')
parser.add_argument('--num_sim', type=int, default=20,
                    help='number of simulations performed in synthetic-comparison.py')
args = parser.parse_args()

def recursive_dd():
    return defaultdict(recursive_dd)

if __name__ == "__main__":
    
    with open(args.inputfile , 'rb') as stuff:
        results = pickle.load(stuff)

    table = recursive_dd()

    # getting the maximum observed values for each test function
    # as outlined on page 17-18 of the paper
    for synthetic_name, synthetic_obj in synthetic_functions.items():
        all_max = []
        for optimizer_name, optimizer in optimizers.items():
            N = len(results[optimizer_name][synthetic_name]) # number of simulations
            for sim in np.arange(N):
                all_max.append(np.max(results[optimizer_name][synthetic_name][sim]['y']))
        synthetic_obj['maximum'] = np.max(all_max)

    # monte carlo estimate of the average value in the domain
    # as outlined on pagge 17-18 of the paper
    num_samples = 10**6

    for synthetic_name, synthetic_obj in synthetic_functions.items():

        bound_mins = np.array([bnd[0] for bnd in synthetic_obj['bnds']])
        bound_maxs = np.array([bnd[1] for bnd in synthetic_obj['bnds']])

        u = np.random.uniform(size=(num_samples, len(synthetic_obj['bnds'])))
        x_samples = u * (bound_maxs - bound_mins) + bound_mins

        y_samples = synthetic_obj['func'](x_samples.T)
        synthetic_obj['avg'] = np.mean(y_samples)


    for synthetic_name, synthetic_obj in synthetic_functions.items():
        for optimizer_name, optimizer in optimizers.items():
            
            N = len(results[optimizer_name][synthetic_name]) # number of simulations
            M = len(results[optimizer_name][synthetic_name][0]['y']) # number of iterations
            cur_array = np.zeros((N, M))
            for sim in np.arange(N):
                cur_array[sim,:] = results[optimizer_name][synthetic_name][sim]['y']

            # compute the target value we are looking for
            # more thought needed here - the maximum is computed using the results
            # according to the paper and the avg is computed by Monte Carlo 
            cur_max = synthetic_obj['maximum']
            cur_avg = synthetic_obj['avg']
            target = cur_max - (cur_max - cur_avg) * (1 - args.target)

            # find the number of iterations it took to reach target
            # TODO: should change the hard coded 1000 to passed argument
            loc_pass_target = np.zeros(N)
            for i in range(N):
                above = cur_array[i] >= target
                if not np.any(above):
                    loc_pass_target[i] = args.num_sim
                else:
                    loc_pass_target[i] = np.min(np.nonzero(above)[0]) # minimum index of nonzero element

            # the below code is broken because it sets the number of iterations to zero in the case
            # that you reach the target on the first iteration
            # (which, shockingly, happens somewhat frequently for these very easy targets)

            # loc_pass_target = np.argmax(cur_array >= target, axis=1) # np.where, np.nonzero
            # loc_pass_target[loc_pass_target == 0] = 1000

            # add the computed results to the dictionary that we will 
            # serialize and/or reformat to csv for writing
            cur_results = {'mean': np.mean(loc_pass_target), 'std': np.std(loc_pass_target)}
            table[optimizer_name][synthetic_name] = cur_results
    
    # serialize
    # note that if you want to load this serialized object you 
    # need to have recursive_dd defined on the other end
    with open(args.outputfile + '.pickle', 'wb') as place:
        pickle.dump(table, place, protocol=pickle.HIGHEST_PROTOCOL)

    csv_table = recursive_dd()
    for optimizer_name, contents in table.items():
        for func_name, results in contents.items():
            csv_table[optimizer_name][func_name] = str(results['mean']) + ' +/- {0:.01f}'.format(results['std'])

    df = pd.DataFrame.from_dict(csv_table).T
    df = df[['Holder Table', 'Rosenbrock', 'Linear Slope', 'Sphere', 'Deb N.1']]

    df.to_csv(args.outputfile + '.csv')
    