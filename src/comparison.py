#!usr/bin/env python

# Script to generate a loss versus iterations plot
# for lipo and pure random search
#
# Very limited at this point, but its a start
#
# Usage: python comparison.py filename='imagename.png' [--num_sim, --num_iter]

from sequential import lipo, pure_random_search, adaptive_lipo
from plotting import loss_v_iter

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str)
parser.add_argument('--function', type=str)
parser.add_argument('--num_sim', type=int, default=20)
parser.add_argument('--num_iter', type=int, default=100)
args = parser.parse_args()

#---------------------------------------------#
# THE BOUNDS AND FUNCTION SHOULD BE ARGUMENTS #
#---------------------------------------------#

if args.function == "holder_table":

    def f(x):
        inside_exp = np.abs(1-np.sqrt(x[0]*x[0]+x[1]*x[1])/np.pi)
        return np.abs(np.sin(x[0])*np.cos(x[1])*np.exp(inside_exp))

    #k = 40
    bnds = [(-10,10),(-10,10)]

#----------------------------------------------------#
# THE SEQUENTIAL STRATEGIES SHOULD ALSO BE ARGUMENTS #
#----------------------------------------------------#

def main():

    results_lipo = np.zeros((args.num_sim, args.num_iter))
    results_prs = np.zeros((args.num_sim, args.num_iter))

    for sim in np.arange(args.num_sim):

        lipo_output = adaptive_lipo(func=f, bounds=bnds, n=args.num_iter)
        prs_output = pure_random_search(func=f, bounds=bnds, n=args.num_iter)

        results_lipo[sim,:] = lipo_output['loss']
        results_prs[sim,:] = prs_output['loss']

    loss_v_iter(
        loss=[results_lipo, results_prs], 
        names=['Adaptive LIPO','PRS'],
        color=['blue', 'orange'], 
        figsize=(20,10), 
        filename=args.filename
    )

if __name__ == "__main__":
    main()