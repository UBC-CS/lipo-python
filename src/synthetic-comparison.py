#!usr/bin/env python

# Script to replicate the results of the numerical 
#
# Very limited at this point, but its a start
#
# Usage: python comparison.py filename='imagename.png' [--num_sim, --num_iter]

from sequential import optimizers
from objective_functions import synthetic_functions

import numpy as np
import argparse
import pickle
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str)
parser.add_argument('--num_sim', type=int, default=20)
parser.add_argument('--num_iter', type=int, default=100)
args = parser.parse_args()

def main():

    results = recursive_dd()

    # loop over our sequential algortihms
    for optimizer_name, optimizer in optimizers.items():
        # loop over the objective functions
        for synthetic_name, synthetic_obj in synthetic_functions.items():
            # perform specified number of simulations
            for sim in np.arange(args.num_sim):
                print(sim)
                out = optimizer(func=synthetic_obj['func'], bounds=synthetic_obj['bnds'], n=args.num_iter)

                results[optimizer_name][synthetic_name][sim] = out
    
    # serialize
    # note that if you want to load this serialized object you 
    # need to have recursive_dd defined on the other end
    with open(args.filename + '.pickle', 'wb') as place:
        pickle.dump(results, place, protocol=pickle.HIGHEST_PROTOCOL)


def recursive_dd():
    return defaultdict(recursive_dd)

if __name__ == "__main__":
    main()
             



