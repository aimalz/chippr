import numpy as np
import os

import matplotlib as mpl
# mpl.use('PS')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import chippr
from chippr import defaults as d
from chippr import utils as u
from chippr import plot_utils as pu

def plot_true_histogram(true_samps, n_bins=(10, 50), plot_loc='', prepend='', plot_name='true_hist.png'):
    """
    Plots a histogram of true input values

    Parameters
    ----------
    true_samps: numpy.ndarray, float
        vector of true values of scalar input
    n_bins: tuple, int, optional
        number of histogram bins in which to place input values, coarse and fine
    plot_loc: string, optional
        location in which to store plot
    plot_name: string, optional
        filename for plot
    prepend: str, optional
        prepend string to plot name
    """
    pu.set_up_plot()
    f = plt.figure(figsize=(5, 5))
    sps = f.add_subplot(1, 1, 1)
    sps.hist(true_samps, bins=n_bins[1], color='k', alpha=0.5, log=True)#, normed=1
    sps.hist(true_samps, bins=n_bins[0], color='y', alpha=0.5, log=True)#, normed=1
    sps.set_xlabel(r'$z_{true}$')
    sps.set_ylabel(r'$n(z_{true})$')
    f.savefig(os.path.join(plot_loc, prepend+plot_name), bbox_inches='tight', pad_inches = 0, dpi=d.dpi)

    return

def plot_prob_space(z_grid, p_space, plot_loc='', prepend='', plot_name='prob_space.png'):
    """
    Plots the 2D probability space of z_spec, z_phot

    Parameters
    ----------
    p_space: numpy.ndarray, float
        probabilities on the grid
    z_grid: numpy.ndarray, float
        fine grid of redshifts
    plot_loc: string, optional
        location in which to store plot
    plot_name: string, optional
        filename for plot
    prepend: str, optional
        prepend string to plot name
    """
    pu.set_up_plot()
    f = plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    grid_len = len(z_grid)
    grid_range = range(grid_len)
    # to_plot = u.safe_log(p_space.evaluate(all_points.reshape((grid_len**2, 2))).reshape((grid_len, grid_len)))
    # to_plot.reshape((len(z_grid), len(z_grid)))
    # plt.pcolormesh(z_grid, z_grid, to_plot, cmap='viridis')
    all_points = np.array([[(z_grid[kk], z_grid[jj]) for kk in grid_range] for jj in grid_range])
    orig_shape = np.shape(all_points)
    # all_vals = np.array([[p_space.evaluate_one(np.array([z_grid[jj], z_grid[kk]])) for jj in range(len(z_grid))] for kk in range(len(z_grid))])
    all_vals = p_space.evaluate(all_points.reshape((orig_shape[0]*orig_shape[1], orig_shape[2]))).reshape((orig_shape[0], orig_shape[1]))
    plt.pcolormesh(z_grid, z_grid, u.safe_log(all_vals), cmap='viridis')
    plt.plot(z_grid, z_grid, color='k')
    plt.colorbar()
    debugging = p_space.sample(1000).T
    np.savetxt(os.path.join(plot_loc, 'int_pr_samps.txt'), debugging)
    plt.scatter(debugging[0], debugging[1], s=1, color='k', marker='.')
    plt.xlabel(r'$z_{\mathrm{true}}$')
    plt.ylabel(r'$\mathrm{``data"}$')#z_{\mathrm{phot}}$')
    plt.axis([z_grid[0], z_grid[-1], z_grid[0], z_grid[-1]])
    f.savefig(os.path.join(plot_loc, prepend+plot_name), bbox_inches='tight', pad_inches = 0, dpi=d.dpi)
    return

def plot_mega_scatter(zs, pfs, z_grid, grid_ends, truth=None, plot_loc='', prepend='', plot_name='mega_scatter.png', int_pr=None):
    """
    Plots a scatterplot of true and observed redshift values

    Parameters
    ----------
    zs: numpy.ndarray, float
        matrix of spec, phot values
    z_grid: numpy.ndarray, float
        fine grid of redshifts
    grid_ends: numpy.ndarray, float
        coarse bin ends
    pfs: numpy.ndarray, float
        matrix of posteriors evaluated on a fine grid
    truth: numpy.ndarray, float
        (x, y) coordinates of the true distribution on a fine grid
    plot_loc: string, optional
        location in which to store plot
    plot_name: string, optional
        filename for plot
    prepend: str, optional
        prepend string to plot name
    int_pr: numpy.ndarray, float, optional
        plot the interim prior with the histograms?
    """
    print('debug demo')
    print((truth, np.shape(zs), np.shape(pfs), np.shape(z_grid), np.shape(grid_ends)))

    n = len(zs)
    zs = zs.T
    true_zs = zs[0]
    obs_zs = zs[1]

    pu.set_up_plot()
    f, scatplot = plt.subplots(figsize=(7.5, 7.5))
    f.subplots_adjust(hspace=0)

    # true_hist = np.hist(true_zs, grid_ends)
    # sps_x.step(true_hist[0], true_hist[1], c='k', where='mid')
    # sps_x.hist(obs_zs, bins=info['bin_ends'][0])
    # sps_y.hist(true_zs)

    spacing = np.mean(z_grid[1:] - z_grid[:-1])

    scatplot.plot(z_grid, z_grid, color='k', alpha=0.5, linewidth=1.5)
    scatplot.scatter(true_zs, obs_zs, c='k', marker='.', s=1., alpha=0.1)
    randos = np.floor(n / (d.plot_colors + 1)) * np.arange(1., d.plot_colors + 1)# np.random.choice(range(len(z_grid)), d.plot_colors)
    randos = randos.astype(int)
    max_pfs = np.max(pfs)
    sort_inds = np.argsort(obs_zs)
    sorted_pfs = pfs[sort_inds]
    sorted_true = true_zs[sort_inds]
    sorted_obs = obs_zs[sort_inds]
    for r in range(d.plot_colors):
        pf = sorted_pfs[randos[r]]
        norm_pf = pf / max_pfs
        pu.plot_h(scatplot, [min(grid_ends), max(grid_ends)], [sorted_obs[randos[r]], sorted_obs[randos[r]]], c='k', s=':', w=0.5)
        pu.plot_v(scatplot, [min(grid_ends), sorted_true[randos[r]], max(z_grid)], [sorted_obs[randos[r]], max(norm_pf)+sorted_obs[randos[r]]], c=pu.colors[r], s=':', w=1.)
        scatplot.step(z_grid, norm_pf + sorted_obs[randos[r]], c=pu.colors[r], where='mid')# plt.plot(z_grid, norm_pf + sorted_obs[randos[r]], c='k')
    limval = (max(grid_ends) - min(grid_ends)) / (len(grid_ends) - 1.)
    scatplot.set_xlim([min(grid_ends)-limval, max(grid_ends)+limval])
    scatplot.set_ylim([min(grid_ends)-limval, max(grid_ends)+limval])
    scatplot.set_xticks(np.linspace(np.floor(min(grid_ends)), np.ceil(max(grid_ends)), 5))
    scatplot.set_yticks(np.linspace(np.floor(min(grid_ends)), np.ceil(max(grid_ends)), 5))
    scatplot.set_xlabel(r'$z_{true}$')
    scatplot.set_ylabel(r'$z_{est}$')
    scatplot.text(0.25, 3., r'mock $p(z\mid \mathrm{``data"})$', rotation=0, size=20)

    # scatplot.set_aspect(1.)
    divider = make_axes_locatable(scatplot)
    histx = divider.append_axes('top', 1.2, pad=0., sharex=scatplot)
    histy = divider.append_axes('right', 1.2, pad=0., sharey=scatplot)

    histx.hist(true_zs, bins=grid_ends, alpha=0.25, color='k', stacked=False, density=True)
    histy.hist(obs_zs, bins=grid_ends, orientation='horizontal', alpha=0.25, color='k', stacked=False, density=True)
    if truth is not None:
        histx.step(truth[0], truth[1], c='k', where='mid')
        pu.plot_step(histy, np.pad(truth[1], (1, 1), 'constant', constant_values=(0, 0)), u.ends_from_mids(truth[0]), c='k', w=1.5)
    if int_pr is not None:
        histx.step(int_pr[0], int_pr[1], c='k', alpha=0.5, where='mid')
        pu.plot_step(histy, np.pad(int_pr[1], (1, 1), 'constant', constant_values=(0, 0)), u.ends_from_mids(int_pr[0]), c='k', a=0.5, w=1.5)
    histx.xaxis.set_tick_params(labelbottom=False)
    histy.yaxis.set_tick_params(labelleft=False)
    histx.set_yticks([])
    histy.set_xticks([])
    histx.set_xlim([min(grid_ends)-limval, max(grid_ends)+limval])
    histy.set_ylim([min(grid_ends)-limval, max(grid_ends)+limval])

    f.savefig(os.path.join(plot_loc, prepend+plot_name), bbox_inches='tight', pad_inches=0, dpi=d.dpi)
    return

def plot_scatter(zs, pfs, z_grid, plot_loc='', prepend='', plot_name='scatter.png'):
    """
    Plots a scatterplot of true and observed redshift values

    Parameters
    ----------
    zs: numpy.ndarray, float
        matrix of spec, phot values
    z_grid: numpy.ndarray, float
        fine grid of redshifts
    pfs: numpy.ndarray, float
        matrix of posteriors evaluated on a fine grid
    plot_loc: string, optional
        location in which to store plot
    plot_name: string, optional
        filename for plot
    prepend: str, optional
        prepend string to plot name
    """
    n = len(zs)
    zs = zs.T
    true_zs = zs[0]
    obs_zs = zs[1]
    pu.set_up_plot()
    f = plt.figure(figsize=(5, 5))
    sps = f.add_subplot(1, 1, 1)
    sps.plot(z_grid, z_grid, color='r', alpha=0.5, linewidth=2.)
    sps.scatter(true_zs, obs_zs, c='g', marker='.', s = 1., alpha=0.1)
    randos = np.floor(n / (d.plot_colors + 1)) * np.arange(1., d.plot_colors + 1)# np.random.choice(range(len(z_grid)), d.plot_colors)
    randos = randos.astype(int)
    max_pfs = np.max(pfs)
    sort_inds = np.argsort(obs_zs)
    sorted_pfs = pfs[sort_inds]
    sorted_true = true_zs[sort_inds]
    sorted_obs = obs_zs[sort_inds]
    for r in range(d.plot_colors):
        pf = sorted_pfs[randos[r]]
        norm_pf = pf / max_pfs / (d.plot_colors + 1)
        plt.step(z_grid, norm_pf + sorted_obs[randos[r]], c='k', where='mid')# plt.plot(z_grid, norm_pf + sorted_obs[randos[r]], c='k')
        plt.hlines(sorted_obs[randos[r]], min(z_grid), max(z_grid), color='k', alpha=0.5, linestyle='--')
        plt.scatter(sorted_true[randos[r]], sorted_obs[randos[r]], marker='+', c='r')
    sps.set_xlabel(r'$z_{true}$')
    sps.set_ylabel(r'$z_{obs}$')
    f.savefig(os.path.join(plot_loc, prepend+plot_name), bbox_inches='tight', pad_inches = 0, dpi=d.dpi)
    # plt.close()
    return

def plot_obs_scatter(true_vals, pfs, z_grid, plot_loc='', prepend='', plot_name='obs_scatter.png'):
    """
    Plots a scatterplot of true and observed redshift values

    Parameters
    ----------
    true_vals: numpy.ndarray, float
        2 * n_gals array of true values of scalar input
    pfs: numpy.ndarray, float
        matrix of interim posteriors evaluated on a fine grid
    z_grid: numpy.ndarray, float
        vector of bin midpoints
    plot_loc: string, optional
        location in which to store plot
    plot_name: string, optional
        filename for plot
    prepend: str, optional
        prepend string to plot name
    """
    true_zs = true_vals[0]
    obs_zs = true_vals[1]#np.array([z_grid[np.argmax(pf)] for pf in pfs])
    max_pfs = np.max(pfs)
    n = len(true_zs)
    dz = (max(z_grid) - min(z_grid)) / len(z_grid)
    jitters = np.random.uniform(-1. * dz / 2., dz / 2., n)
    obs_zs = obs_zs + jitters
    pu.set_up_plot()
    f = plt.figure(figsize=(5, 5))
    sps = f.add_subplot(1, 1, 1)
    sps.scatter(true_zs, obs_zs, c='k', marker='.', s = 1., alpha=0.1)
    randos = np.floor(n / (d.plot_colors + 1)) * np.arange(1., d.plot_colors + 1)# np.random.choice(range(len(z_grid)), d.plot_colors)
    randos = randos.astype(int)
    ordered = np.argsort(obs_zs)
    sorted_pfs = pfs[ordered]
    sorted_true = true_zs[ordered]
    sorted_obs = obs_zs[ordered]
    for r in range(d.plot_colors):
        pf = sorted_pfs[randos[r]]
        plt.scatter(sorted_true[randos[r]], sorted_obs[randos[r]], marker='+', c='r')
        norm_pf = pf / max_pfs / (d.plot_colors + 1)
        plt.step(z_grid, norm_pf + sorted_obs[randos[r]], c='k', where='mid')
        plt.hlines(sorted_obs[randos[r]], min(z_grid), max(z_grid), color='k', alpha=0.5, linestyle='--')
    sps.set_xlabel(r'$z_{true}$')
    sps.set_ylabel(r'$\hat{z}_{MAP}$')
    f.savefig(os.path.join(plot_loc, prepend+plot_name), bbox_inches='tight', pad_inches = 0, dpi=d.dpi)

    return
