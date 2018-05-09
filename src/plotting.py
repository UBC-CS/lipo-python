"""
Utility for plotting loss versus number iterations
"""

import numpy as np
import matplotlib.pyplot as plt

def loss_v_iter(loss, color, names, filename=None, q=(5,95), figsize=(10,5)):
    """
    Utility function for plotting loss versus iterations in experiments.
    
    Parameters
    ----------
    loss_arr:  list of numpy arrays each of shape (N, M) where each row is the 
         result of one simulation run of # of columns iterations of algorithm
    color:     color scheme of resulting plot
    names:     the names 
    filename:  string representing the filename of the plot that will be saved
    q:         tuple containing upper and lower percentiles to use in errorbars
    figsize:   tuple containing size of saved image in inches
    
    Returns
    -------
    displays a plot if interactive - saves an image  
    """

    if not isinstance(loss, list):
        raise TypeError('loss should be a list! (even if it is only one element)')
        
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize[0], figsize[1])
    
    for ind, loss_arr in enumerate(loss):

        # extract results necessary for plotting
        num_sim, num_iter = loss_arr.shape
        median_loss = np.median(a=loss_arr, axis=0)
        upper_loss = np.percentile(a=loss_arr,q=q[1], axis=0)
        lower_loss = np.percentile(a=loss_arr,q=q[0], axis=0)
        yerr = np.abs(np.vstack((lower_loss,  upper_loss)) - median_loss)


        # plot the results
        ax.plot(range(1,num_iter+1), median_loss, color=color[ind])
        ax.errorbar(
            x=range(1,num_iter+1), 
            y=median_loss,
            yerr=yerr, 
            linestyle='None',
            alpha=0.5, 
            capsize=200/num_iter,
            color=color[ind]
        )
        ax.set(xlabel='Iteration Number', ylabel='Error')

    plt.legend(names)
    if filename:
        fig = ax.get_figure()
        fig.savefig(filename)
    else:
        plt.show();