def check_extra_params(params):
    """
    Sets parameter values pertaining to true n(z)

    Parameters
    ----------
    params: dict
        dictionary containing key/value pairs for simulation

    Returns
    -------
    params: dict
        dictionary containing key/value pairs for simulation
    """
    if 'smooth_truth' not in params:
        params['smooth_truth'] = 0
    else:
        params['smooth_truth'] = int(params['smooth_truth'][0])

    if 'interim_prior' not in params:
        params['interim_prior'] = 'flat'
    else:
        params['interim_prior'] = str(params['interim_prior'][0])
    if 'n_galaxies' not in params:
        params['n_galaxies'] = 10**4
    else:
        params['n_galaxies'] = 10**int(params['n_galaxies'][0])

    return(params)

def make_true(given_key):
    """
    Function to create true redshift distribution to be shared among several
    test cases

    Parameters
    ----------
    given_key: string
        name of test case for which true n(z) is to be made

    Returns
    -------
    true_nz: chippr.gmix object
        gaussian mixture probability distribution

    Notes
    -----
    test_name is currently ignored but will soon be used to load parameters for
    making true_nz instead of hardcoded values.
    """
    test_info = all_tests[given_key]

    if test_info['params']['smooth_truth'] == 1:
        true_amps = np.array([0.150,0.822,1.837,2.815,3.909,
                              5.116,6.065,6.477,6.834,7.304,
                              7.068,6.771,6.587,6.089,5.165,
                              4.729,4.228,3.664,3.078,2.604,
                              2.130,1.683,1.348,0.977,0.703,
                              0.521,0.339,0.283,0.187,0.141,
                              0.104,0.081,0.055,0.043,0.034])
        true_grid = np.linspace(test_info['bin_ends'][0], test_info['bin_ends'][-1], len(true_amps) + 1)
        true_grid_mids = (true_grid[1:] + true_grid[:-1]) / 2.
        f = spi.interp1d(true_grid_mids, true_amps)
        bin_mids = (test_info['bin_ends'][1:] + test_info['bin_ends'][:-1]) / 2.
        bin_difs = test_info['bin_ends'][1:] - test_info['bin_ends'][:-1]
        true_means = bin_mids
        true_amps = f(bin_mids)
        true_sigmas = bin_difs
    else:
        bin_range = max(test_info['bin_ends']) - min(test_info['bin_ends'])
        true_amps = np.array([0.20, 0.35, 0.55])
        true_means = np.array([0.5, 0.2, 0.75]) * bin_range + min(test_info['bin_ends'])
        true_sigmas = np.array([0.4, 0.2, 0.1]) * bin_range

    n_mix_comps = len(true_amps)
    true_funcs = []
    for c in range(n_mix_comps):
        true_funcs.append(chippr.gauss(true_means[c], true_sigmas[c]**2))
    true_nz = chippr.gmix(true_amps, true_funcs,
            limits=(min(test_info['bin_ends']), max(test_info['bin_ends'])))

    true_dict = {'amps': true_amps, 'means': true_means, 'sigmas': true_sigmas}
    true_dict['bins'] = test_info['bin_ends']

    # true_zs = true_nz.sample(test_info['params']['n_galaxies'])
    # true_dict['zs'] = true_zs
    test_info['truth'] = true_dict

    return(test_info)

def make_interim_prior(given_key):
    """
    Function to make the histogram-parametrized interim prior

    Parameters
    ----------
    given_key: string
        name of test case for which interim prior is to be made

    Returns
    -------
    interim_prior: chippr.discrete or chippr.gauss or chippr.gmix object
        the discrete distribution that will be the interim prior
    """
    test_info = all_tests[given_key]

    if test_info['params']['interim_prior'] == 'template':
        bin_range = max(test_info['bin_ends']) - min(test_info['bin_ends'])
        int_amps = np.array([0.35, 0.5, 0.15])
        int_means = np.array([0.1, 0.5, 0.9]) * bin_range + min(test_info['bin_ends'])
        int_sigmas = np.array([0.1, 0.1, 0.1]) * bin_range
        n_mix_comps = len(int_amps)
        int_funcs = []
        for c in range(n_mix_comps):
            int_funcs.append(chippr.gauss(int_means[c], int_sigmas[c]**2))
        interim_prior = chippr.gmix(int_amps, int_funcs,
            limits=(min(test_info['bin_ends']), max(test_info['bin_ends'])))
    elif test_info['params']['interim_prior'] == 'training':
        int_amps = np.array([0.150,0.822,1.837,2.815,3.909,
                              5.116,6.065,6.477,6.834,7.304,
                              7.068,6.771,6.587,6.089,5.165,
                              4.729,4.228,3.664,3.078,2.604,
                              2.130,1.683,1.348,0.977,0.703,
                              0.521,0.339,0.283,0.187,0.141,
                              0.104,0.081,0.055,0.043,0.034])
        int_grid = np.linspace(test_info['bin_ends'][0], test_info['bin_ends'][-1], len(int_amps) + 1)
        int_grid_mids = (int_grid[1:] + int_grid[:-1]) / 2.
        f = spi.interp1d(int_grid_mids, int_amps)
        bin_mids = (test_info['bin_ends'][1:] + test_info['bin_ends'][:-1]) / 2.
        bin_difs = test_info['bin_ends'][1:] - test_info['bin_ends'][:-1]
        int_amps = f(bin_mids)
        int_means = bin_mids
        int_sigmas = bin_difs
        n_mix_comps = len(int_amps)
        int_funcs = []
        for c in range(n_mix_comps):
            int_funcs.append(chippr.gauss(int_means[c], int_sigmas[c]**2))
        interim_prior = chippr.gmix(int_amps, int_funcs,
                limits=(min(test_info['bin_ends']), max(test_info['bin_ends'])))
    else:
        bin_ends = np.array([test_info['params']['bin_min'], test_info['params']['bin_max']])
        weights = np.array([1.])
        interim_prior = chippr.discrete(bin_ends, weights)

    return(interim_prior)

def make_catalog(given_key):
    """
    Function to create a catalog once true redshifts exist

    Parameters
    ----------
    given_key: string
        name of test case to be run
    """
    test_info = all_tests[given_key]
    test_name = test_info['name']

    test_dir = os.path.join(result_dir, test_name)
    test_info['dir'] = test_dir
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    param_file_name = test_name + '.txt'
    params = chippr.utils.ingest(param_file_name)
    params = defaults.check_sim_params(params)
    params = check_extra_params(params)
    test_info['params'] = params

    test_info['bin_ends'] = np.linspace(test_info['params']['bin_min'],
                                test_info['params']['bin_max'],
                                test_info['params']['n_bins'] + 1)

    test_info = make_true(given_key)
    true_amps = test_info['truth']['amps']
    true_means = test_info['truth']['means']
    true_sigmas =  test_info['truth']['sigmas']
    n_mix_comps = len(true_amps)
    true_funcs = []
    for c in range(n_mix_comps):
        true_funcs.append(chippr.gauss(true_means[c], true_sigmas[c]**2))
    true_nz = chippr.gmix(true_amps, true_funcs,
            limits=(test_info['params']['bin_min'], test_info['params']['bin_max']))

    interim_prior = make_interim_prior(given_key)

    posteriors = chippr.catalog(param_file_name, loc=test_dir)
    output = posteriors.create(true_nz, interim_prior, N=test_info['params']['n_gals'])
    # data = np.exp(output['log_interim_posteriors'])
    posteriors.write()
    data_dir = posteriors.data_dir
    with open(os.path.join(data_dir, 'true_params.p'), 'w') as true_file:
        pickle.dump(test_info['truth'], true_file)

if __name__ == "__main__":

    import numpy as np
    import pickle
    import shutil
    import os
    import scipy.interpolate as spi

    import chippr
    from chippr import *

    result_dir = os.path.join('..', 'results')
    test_name = 'fiducial'
    all_tests = {}
    test_info = {}
    test_info['name'] = test_name
    all_tests[test_name] = test_info

    make_catalog(test_name)
