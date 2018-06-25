#!/usr/bin/env python

# Usage:
#    python figure_five.py inputfile outputfile
#    optional args: [--target, --raw]

from sequential import optimizers
from objective_functions import objectives

import numpy as np
import argparse
import pickle
import pandas as pd
from collections import defaultdict

def compute_average(objectives, results, n_samples=10**6):
    """Compute the average value of the provided 
    objective functions using a monte carlo estimate
    
    Input dictionary `objectives` must be of the same 
    form as `objective_functions.objectives`
    
    Input results must be output from `optimize.py`"""

    for func_name in results.keys():

        objective = objectives[func_name]

        bound_mins = np.array([bnd[0] for bnd in objective['bnds']])
        bound_maxs = np.array([bnd[1] for bnd in objective['bnds']])

        u = np.random.uniform(size=(n_samples, len(objective['bnds'])))
        x_samples = u * (bound_maxs - bound_mins) + bound_mins

        # the following line works for the synthetic objective functions
        # but I doubt that it will work for the 'real world' examples
        y_samples = objective['func'](x_samples.T)
        objectives[func_name]['avg'] = np.mean(y_samples)
    
    return objectives

def compute_maximum(objectives, results):
    """Compute the maximum value of the provided 
    objective functions by looking at maximum across 
    all simulations
    
    Input dictionary `objectives` must be of the same 
    form as `objective_functions.objectives`
    
    Input results must be output from `optimize.py`"""

    for func_name, func_results in results.items():
        all_max = []
        for optimizer_name, opt_results in func_results.items():
            N = len(opt_results) # number of simulations
            for sim in np.arange(N):
                all_max.append(np.max(opt_results[sim]['y']))
        objectives[func_name]['maximum'] = np.max(all_max)
    
    return objectives

def table_to_df(table):
    """Utility to convert table dictionary to dataframe"""

    colnames = list(table.keys())
    
    csv_table = defaultdict(lambda: defaultdict(dict))
    for func_name, contents in table.items():
        #csv_table[func_name] = dict()
        for optimizer_name, opt_results in contents.items():
            #csv_table[func_name][optimizer_name] = dict()
            for target, results in opt_results.items():
                cell = str(results['mean']) + ' +/- {0:.01f}'.format(results['std'])
                csv_table[target][optimizer_name][func_name] = cell
    
    multi_ind = {(i,j): csv_table[i][j] for i in csv_table.keys() 
                                        for j in csv_table[i].keys()}

    df = pd.DataFrame.from_dict(multi_ind).T
    df = df[colnames]
    return df

def main(results, objectives, target):

    objectives = compute_average(objectives, results)
    objectives = compute_maximum(objectives, results)

    table = dict()

    for func_name, func_results in results.items():
        
        table[func_name] = dict()
        for optimizer_name, opt_results in func_results.items():
            
            N = len(opt_results)         # number of simulations
            M = len(opt_results[0]['y']) # number of iterations

            # populate the array of simulation results 
            cur_array = np.zeros((N, M))
            for sim in np.arange(N):
                cur_array[sim,:] = opt_results[sim]['y']

            cur_max = objectives[func_name]['maximum']
            cur_avg = objectives[func_name]['avg']

            cur_results = dict()
            for each_target in target:

                cur_target = cur_max - (cur_max - cur_avg) * (1 - each_target)

                # find the first position at which we reach the target value. 
                # if we don't, set it to number of simulations
                loc_pass_target = np.zeros(N)
                for i in range(N):
                    above = cur_array[i] >= cur_target
                    if not np.any(above):
                        loc_pass_target[i] = M
                    else:
                        loc_pass_target[i] = np.min(np.nonzero(above)[0])
                
                cur_results[each_target] = {'mean': np.mean(loc_pass_target), 'std': np.std(loc_pass_target)}

            # store the results          
            table[func_name][optimizer_name] = cur_results

    return table
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', type=str,
                        help='input file (results from running optimize.py)')
    parser.add_argument('outputfile', type=str,
                        help='outputfile')
    parser.add_argument('--target', type=float,
                        help='''float between 0 and 1 indicating target value(s) 
                                we seek to reach in optimization process''',
                        choices=[0.9, 0.95, 0.99])
    parser.add_argument('--raw', default=False, action='store_true',
                        help='indicate if we want a csv output or raw pickle')
    args = parser.parse_args()

    with open(args.inputfile , 'rb') as stuff:
        results = pickle.load(stuff)
    
    if args.target:
        target = [args.target]
    else:
        target = [0.9, 0.95, 0.99]

    table = main(results, objectives, target)

    if args.raw:
        with open(args.outputfile + '.pkl', 'wb') as place:
            pickle.dump(table, place, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        df = table_to_df(table)
        df.to_csv(args.outputfile + '.csv')
