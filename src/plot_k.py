#!usr/bin/env python

# Script to plot the lipschitz estimates
# Usage:
#    - python plot_k.py inputfile 1 myplot

import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(results, lipschitz_constant, filename, 
         optimizer='AdaLIPO', q=(5,95), figsize=(10,5)):

    if len(results.keys()) > 1:
        raise RuntimeError('Inputfile must be simulation using single objective function!')
    
    func_name = list(results.keys())[0] 
    d = results[func_name][optimizer][0]['x'].shape[1]
    num_sim = len(results[func_name][optimizer])
    num_iter = len(results[func_name][optimizer][0]['k'])

    k_results = np.zeros((num_sim, num_iter))

    for sim in range(num_sim):
        k_results[sim,:] = results[func_name][optimizer][sim]['k']

    median_loss = np.median(a=k_results, axis=0)
    upper_loss = np.percentile(a=k_results, q=q[1], axis=0)
    lower_loss = np.percentile(a=k_results, q=q[0], axis=0)
    yerr = np.abs(np.vstack((lower_loss,  upper_loss)) - median_loss)

    fig, ax = plt.subplots()
    fig.set_size_inches(figsize[0], figsize[1])

    ax.plot(range(1,num_iter+1), [lipschitz_constant]*num_iter, color='red')

    ax.plot(range(1,num_iter+1), median_loss)
    ax.errorbar(
        x=range(1,num_iter+1), 
        y=median_loss,
        yerr=yerr, 
        linestyle='None',
        alpha=0.5, 
        capsize=200/num_iter
    )
    ax.set(xlabel='Iteration Number', ylabel='Lipschitz Constant')

    plt.legend(['True', 'Estimated', '90 % Error bars'])
    plt.title('Convergence of Lipschitz Constant Estimate of {}-d Paraboloid'.format(d))
    if filename:
        fig = ax.get_figure()
        fig.savefig(filename)
    else:
        plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', type=str)
    parser.add_argument('--K', type=float, default=None)
    parser.add_argument('--filename', type=str, default=None)
    args = parser.parse_args()

    with open(args.inputfile, 'rb') as f:
        results = pickle.load(f)

    main(results, args.K, args.filename)
