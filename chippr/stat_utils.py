import numpy as np
import scipy as sp
from scipy import stats

import chippr
from chippr import defaults as d
from chippr import utils as u

def mean(population):
    """
    Calculates the mean of a population

    Parameters
    ----------
    population: np.array, float
        population over which to calculate the mean

    Returns
    -------
    mean: np.array, float
        mean value over population
    """
    shape = np.shape(population)
    flat = population.reshape(np.prod(shape[:-1]), shape[-1])
    mean = np.mean(flat, axis=0)
    return mean

def norm_fit(population):
    """
    Calculates the mean and standard deviation of a population

    Parameters
    ----------
    population: np.array, float
        population over which to calculate the mean

    Returns
    -------
    norm_stats: tuple, list, float
        mean and standard deviation over population
    """
    shape = np.shape(population)
    flat = population.reshape(np.prod(shape[:-1]), shape[-1]).T
    locs, scales = [], []
    for k in range(shape[-1]):
        (loc, scale) = sp.stats.norm.fit_loc_scale(flat[k])
        locs.append(loc)
        scales.append(scale)
    locs = np.array(locs)
    scales = np.array(scales)
    norm_stats = (locs, scales)
    return norm_stats

def calculate_kld(pe, qe, vb=True):
    """
    Calculates the Kullback-Leibler Divergence between two PDFs.

    Parameters
    ----------
    pe: numpy.ndarray, float
        probability distribution evaluated on a grid whose distance from `q`
        will be calculated.
    qe: numpy.ndarray, float
        probability distribution evaluated on a grid whose distance to `p` will
        be calculated.
    vb: boolean
        report on progress to stdout?

    Returns
    -------
    Dpq: float
        the value of the Kullback-Leibler Divergence from `q` to `p`
    """
    # Normalize the evaluations, so that the integrals can be done
    # (very approximately!) by simple summation:
    pn = pe / np.sum(pe)
    qn = qe / np.sum(qe)
    # Compute the log of the normalized PDFs
    logp = u.safe_log(pn)
    logq = u.safe_log(qn)
    # Calculate the KLD from q to p
    Dpq = np.sum(pn * (logp - logq))
    return Dpq

def calculate_rms(pe, qe, vb=True):
    """
    Calculates the Root Mean Square Error between two PDFs.

    Parameters
    ----------
    pe: numpy.ndarray, float
        probability distribution evaluated on a grid whose distance _from_ `q`
        will be calculated.
    qe: numpy.ndarray, float
        probability distribution evaluated on a grid whose distance _to_ `p`
        will be calculated.
    vb: boolean
        report on progress to stdout?

    Returns
    -------
    rms: float
        the value of the RMS error between `q` and `p`
    """
    npoints = len(pe)
    assert len(pe) == len(qe)
    # Calculate the RMS between p and q
    rms = np.sqrt(np.sum((pe - qe) ** 2) / npoints)
    return rms

def single_parameter_gr_stat(chain):
    """
    Calculates the Gelman-Rubin test statistic of convergence of an MCMC chain
    over one parameter

    Parameters
    ----------
    chain: numpy.ndarray, float
        single-parameter chain

    Returns
    -------
    R_hat: float
        potential scale reduction factor
    """
    ssq = np.var(chain, axis=1, ddof=1)
    W = np.mean(ssq, axis=0)
    xb = np.mean(chain, axis=1)
    xbb = np.mean(xb, axis=0)
    m = chain.shape[0]
    n = chain.shape[1]
    B = n / (m - 1.) * np.sum((xbb - xb)**2., axis=0)
    var_x = (n - 1.) / n * W + 1. / n * B
    R_hat = np.sqrt(var_x / W)
    return R_hat

def multi_parameter_gr_stat(sample):
    """
    Calculates the Gelman-Rubin test statistic of convergence of an MCMC chain
    over multiple parameters

    Parameters
    ----------
    sample: numpy.ndarray, float
        multi-parameter chain output

    Returns
    -------
    Rs: numpy.ndarray, float
        vector of the potential scale reduction factors
    """
    dims = np.shape(sample)
    (n_walkers, n_iterations, n_params) = dims
    n_burn_ins = int(n_iterations / 2)
    chain_ensemble = np.swapaxes(sample, 0, 1)
    chain_ensemble = chain_ensemble[n_burn_ins:, :]
    Rs = np.zeros((n_params))
    for i in range(n_params):
        chains = chain_ensemble[:, :, i].T
        Rs[i] = single_parameter_gr_stat(chains)
    return Rs

def gr_test(sample, threshold=d.gr_threshold):
    """
    Performs the Gelman-Rubin test of convergence of an MCMC chain

    Parameters
    ----------
    sample: numpy.ndarray, float
        chain output
    threshold: float, optional
        Gelman-Rubin test statistic criterion (usually around 1)

    Returns
    -------
    test_result: boolean
        True if burning in, False if post-burn in
    """
    gr = multi_parameter_gr_stat(sample)
    print(('Gelman-Rubin test statistic = '+str(gr)))
    test_result = np.max(gr) > threshold
    return test_result

def cft(xtimes, lag):#xtimes has ntimes elements
    """
    Helper function to calculate autocorrelation time for chain of MCMC samples

    Parameters
    ----------
    xtimes: numpy.ndarray, float
        single parameter values for a single walker over all iterations
    lag: int
        maximum lag time in number of iterations

    Returns
    -------
    ans: numpy.ndarray, float
        autocorrelation time for one time lag for one parameter of one walker
    """
    lent = len(xtimes) - lag
    allt = range(lent)
    ans = np.array([xtimes[t+lag] * xtimes[t] for t in allt])
    return ans

def cf(xtimes):#xtimes has ntimes elements
    """
    Helper function to calculate autocorrelation time for chain of MCMC samples

    Parameters
    ----------
    xtimes: numpy.ndarray, float
        single parameter values for a single walker over all iterations

    Returns
    -------
    cf: numpy.ndarray, float
        autocorrelation time over all time lags for one parameter of one walker
    """
    cf0 = np.dot(xtimes, xtimes)
    allt = range(int(len(xtimes) / 2))
    cf = np.array([sum(cft(xtimes, lag)[int(len(xtimes) / 2):]) for lag in allt]) / cf0
    return cf

def cfs(x, mode):#xbinstimes has nbins by ntimes elements
    """
    Helper function for calculating autocorrelation time for MCMC chains

    Parameters
    ----------
    x: numpy.ndarray, float
        input parameter values of length number of iterations by number of
        walkers if mode='walkers' or dimension of parameters if mode='bins'
    mode: string
        'bins' for one autocorrelation time per parameter, 'walkers' for one
        autocorrelation time per walker

    Returns
    -------
    cfs: numpy.ndarray, float
        autocorrelation times for all walkers if mode='walkers' or all
        parameters if mode='bins'
    """
    if mode == 'walkers':
        xbinstimes = x
        cfs = np.array([sum(cf(xtimes)) for xtimes in xbinstimes]) / len(xbinstimes)
    if mode == 'bins':
        xwalkerstimes = x
        cfs = np.array([sum(cf(xtimes)) for xtimes in xwalkerstimes]) / len(xwalkerstimes)
    return cfs

def acors(xtimeswalkersbins, mode='bins'):
    """
    Calculates autocorrelation time for MCMC chains

    Parameters
    ----------
    xtimeswalkersbins: numpy.ndarray, float
        emcee chain values of dimensions (n_iterations, n_walkers, n_parameters)
    mode: string, optional
        'bins' for one autocorrelation time per parameter, 'walkers' for one
        autocorrelation time per walker

    Returns
    -------
    taus: numpy.ndarray, float
        autocorrelation times by bin or by walker depending on mode
    """
    if mode == 'walkers':
        xwalkersbinstimes = np.swapaxes(xtimeswalkersbins, 1, 2)#nwalkers by nbins by nsteps
        taus = np.array([1. + 2. * sum(cfs(xbinstimes, mode)) for xbinstimes in xwalkersbinstimes])#/len(xwalkersbinstimes)# 1+2*sum(...)
    if mode == 'bins':
        xbinswalkerstimes = xtimeswalkersbins.T#nbins by nwalkers by nsteps
        taus = np.array([1. + 2. * sum(cfs(xwalkerstimes, mode)) for xwalkerstimes in xbinswalkerstimes])#/len(xwalkersbinstimes)# 1+2*sum(...)
    return taus
