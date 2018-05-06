#!usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def loss_v_iter(loss_arr, filename, color='blue', q=(5,95), figsize=(10,5)):
    """
    Utility function for plotting loss versus iterations in experiments.
    
    Parameters
    ----------
    loss_arr:  numpy array of shape (num_sim, num_iter) where each row is the 
         result of one simulation run of # of columns iterations of algorithm
    filename:  string representing the filename of the plot that will be saved
    color:     color scheme of resulting plot
    q:         tuple containing upper and lower percentiles to use in errorbars
    figsize:   tuple containing size of saved image in inches
    
    Returns
    -------
    None - saves an image  
    """
    # extract results necessary for plotting
    num_sim, num_iter = loss_arr.shape
    median_loss = np.median(a=loss_arr, axis=0)
    upper_loss = np.percentile(a=loss_arr,q=q[1], axis=0)
    lower_loss = np.percentile(a=loss_arr,q=q[0], axis=0)
    yerr = np.abs(np.vstack((lower_loss,  upper_loss)) - median_loss)
    
    # plot the results
    plt.figure(figsize=figsize)
    plt.plot(range(1,num_iter+1), median_loss, color=color)
    plt.errorbar(
        x=range(1,num_iter+1), 
        y=median_loss,
        yerr=yerr, 
        linestyle='None',
        alpha=0.3, 
        capsize=200/num_iter, 
        color=color
    )
    plt.xlabel('Iteration Number')
    plt.ylabel('Loss')
    plt.savefig(filename);