#!/usr/bin/env python

from sequential import optimizers
from objective_functions import synthetic_functions

import numpy as np
import argparse
import pickle
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str)
parser.add_argument('--target', type=float)
args = parser.parse_args()

def main():

    with open(args.filename + '.pickle', 'rb') as stuff:
        results = pickle.load(stuff)

    table = recursive_dd()

    # loop over our sequential algortihms
    for optimizer_name, optimizer in optimizers.items():
        # loop over the objective functions
        for synthetic_name, synthetic_obj in synthetic_functions.items():
            
            N = len(results[optimizer_name][synthetic_name])
            M = len(results[optimizer_name][synthetic_name][0]['y'])
            cur_array = np.zeros((N, M))
            for sim in np.arange(N):
                cur_array[sim,:] = results[optimizer_name][synthetic_name][0]['y']

        # compute the target value we are looking for
        # more thought needed here - the maximum is computed using the results
        # according to the paper and the avg is computed by Monte Carlo 
        cur_max = synthetic_obj['maximum']
        cur_avg = synthetic_obj['avg']
        target = cur_max - (cur_max - cur_avg) * args.target

        # find the number of iterations it took to reach target
        # note: shouldchange the hard coded 1000 to passed argument
        loc_pass_target = np.argmax(cur_array >= target, axis=1)
        loc_pass_target[loc_pass_target == 0] = 1000

        # add the computed results to the dictionary that we will 
        # serialize and/or reformat to csv for writing
        cur_results = {'mean': np.mean(loc_pass_target), 'std': np.std(loc_pass_target)}
        table[optimizer_name][synthetic_name] = cur_results
    
    # serialize
    # note that if you want to load this serialized object you 
    # need to have recursive_dd defined on the other end
    #with open(args.filename + '.pickle', 'wb') as place:
    #    pickle.dump(results, place, protocol=pickle.HIGHEST_PROTOCOL)


def recursive_dd():
    return defaultdict(recursive_dd)

if __name__ == "__main__":
    main()