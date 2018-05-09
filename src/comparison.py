#!usr/bin/env python

# Script to generate a loss versus iterations plot
# for lipo and pure random search
#
# Very limited at this point, but its a start
#
# Usage: python comparison.py filename='imagename.png' [--num_sim, --num_iter]

from sequential import lipo, prs
from plotting import loss_v_iter

from scipy.optimize import minimize
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
parser.add_argument('--num_sim', type=int, default=20)
parser.add_argument('--num_iter', type=int, default=100)
args = parser.parse_args()

#-------------------------------------------------------#
# THE BOUNDS, FUNCTION, AND MINIMUM SHOULD BE ARGUMENTS #
#-------------------------------------------------------#

def g(x):
    return (np.cos(x) + 2*np.cos(np.pi*x) - np.sin(np.pi/2*x))

bnds = [(0,4*np.pi)]
scipy_min = minimize(g, np.array([9]), bounds=bnds).fun

#----------------------------------------------------#
# THE SEQUENTIAL STRATEGIES SHOULD ALSO BE ARGUMENTS #
#----------------------------------------------------#

def main():

    results_lipo = np.zeros((args.num_sim, args.num_iter))
    results_prs = np.zeros((args.num_sim, args.num_iter))

    for sim in np.arange(args.num_sim):

        results_lipo[sim,:] = lipo(func=g, bounds=bnds, k=10, n=args.num_iter, seq_out=True)
        results_prs[sim,:] = prs(func=g, bounds=bnds, n=args.num_iter, seq_out=True)

    loss_lipo = np.abs(results_lipo - scipy_min)
    loss_prs = np.abs(results_prs - scipy_min)

    loss_v_iter(
        loss=[loss_lipo, loss_prs], 
        names=['LIPO','PRS'],
        color=['blue', 'orange'], 
        figsize=(20,10), 
        filename=args.filename
    )

if __name__ == "__main__":
    main()