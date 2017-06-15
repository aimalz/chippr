import numpy as np
import os

import matplotlib as mpl
mpl.use('PS')
import matplotlib.pyplot as plt

import chippr
from chippr import defaults as d
from chippr import utils as u
from chippr import plot_utils as pu

def plot_true_histogram(true_samps, n_bins=50, plot_loc='', plot_name='true_hist.png'):
    """
    Plots a histogram of true input values

    Parameters
    ----------
    true_samps: numpy.ndarray, float
        vector of true values of scalar input
    n_bins: int, optional
        number of histogram bins in which to place input values
    plot_loc: string, optional
        location in which to store plot
    plot_name: string, optional
        filename for plot
    """
    pu.set_up_plot()
    f = plt.figure(figsize=(5, 5))
    sps = f.add_subplot(1, 1, 1)
    sps.hist(true_samps, bins=n_bins, normed=1, color='k')
    sps.set_xlabel(r'$z_{true}$')
    sps.set_ylabel(r'$n(z_{true})$')
    f.savefig(os.path.join(plot_loc, plot_name), bbox_inches='tight', pad_inches = 0, dpi=d.dpi)

    return

def plot_obs_scatter(true_zs, pfs, z_grid, plot_loc='', plot_name='scatter.png'):
    """
    Plots a scatterplot of true and observed redshift values

    Parameters
    ----------
    true_zs: numpy.ndarray, float
        vector of true values of scalar input
    pfs: numpy.ndarray, float
        matrix of interim posteriors evaluated on a fine grid
    z_grid: numpy.ndarray, float
        vector of redshifts at which
    plot_loc: string, optional
        location in which to store plot
    plot_name: string, optional
        filename for plot
    """
    obs_zs = np.array([z_grid[np.argmax(pf)] for pf in pfs])
    max_pfs = np.max(pfs)
    n = len(obs_zs)
    dz = (max(z_grid) - min(z_grid)) / len(z_grid)
    jitters = np.random.uniform(-1. * dz / 2., dz / 2., n)
    obs_zs = obs_zs + jitters
    pu.set_up_plot()
    f = plt.figure(figsize=(5, 5))
    sps = f.add_subplot(1, 1, 1)
    sps.scatter(true_zs, obs_zs, c='k', marker='.', s = 1., alpha=0.1)
    randos = np.floor(n / (d.plot_colors + 1)) * np.arange(1., d.plot_colors + 1)# np.random.choice(range(len(z_grid)), d.plot_colors)
    randos = randos.astype(int)
    sorted_pfs = pfs[np.argsort(obs_zs)]
    sorted_true = true_zs[np.argsort(obs_zs)]
    sorted_obs = obs_zs[np.argsort(obs_zs)]
    for r in range(d.plot_colors):
        pf = sorted_pfs[randos[r]]
        plt.scatter(sorted_true[randos[r]], sorted_obs[randos[r]], marker='+', c=pu.colors[r])
        norm_pf = pf / max_pfs / (d.plot_colors + 1)
        plt.plot(z_grid, norm_pf + sorted_obs[randos[r]], c=pu.colors[r])
        plt.hlines(sorted_obs[randos[r]], min(z_grid), max(z_grid), color=pu.colors[r], alpha=0.5, linestyle='--')
    sps.set_xlabel(r'$z_{true}$')
    sps.set_ylabel(r'$\hat{z}_{MAP}$')
    f.savefig(os.path.join(plot_loc, plot_name), bbox_inches='tight', pad_inches = 0, dpi=d.dpi)

    return

def plot_pdfs():
    """
    Plots a small number of randomly chosen likelihood functions and parametrized posteriors
    """
    return
