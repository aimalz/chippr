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
        # params['true_shape'] = int(params['true_shape'][0])
        params['true_loc'] = float(params['true_loc'][0])
        params['true_scale'] = 1./float(params['true_scale'][0])

    if 'interim_prior' not in params:
        params['interim_prior'] = 'flat'
    else:
        params['interim_prior'] = str(params['interim_prior'][0])
    if 'wrong_prior' not in params:
        params['wrong_prior'] = 0
    else:
        params['wrong_prior'] = int(params['wrong_prior'][0])
        if 'bias_model' not in params:
            params['bias_model'] = 'flat'
        else:
            params['bias_model'] = str(params['bias_model'][0])
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
    test_info: dict
        information necessary to reconstruct true_nz
    true_nz: chippr.gmix object
        gaussian mixture probability distribution

    Notes
    -----
    test_name is currently ignored but will soon be used to load parameters for
    making true_nz instead of hardcoded values.
    """
    test_info = all_tests[given_key]

    if test_info['params']['smooth_truth'] == 1:
        # true_shape = test_info['params']['true_shape']
        true_loc = test_info['params']['true_loc']
        true_scale = test_info['params']['true_scale']
        true_nz = gamma(true_loc, true_scale**2, bounds=(min(test_info['bin_ends']), max(test_info['bin_ends'])))#sps.erlang(true_shape, true_loc, true_scale)
        true_dict = {'amps': [1.], 'means': [true_loc], 'sigmas': [true_scale]}
        # true_amps = np.array([0.150,0.822,1.837,2.815,3.909,
        #                       5.116,6.065,6.477,6.834,7.304,
        #                       7.068,6.771,6.587,6.089,5.165,
        #                       4.729,4.228,3.664,3.078,2.604,
        #                       2.130,1.683,1.348,0.977,0.703,
        #                       0.521,0.339,0.283,0.187,0.141,
        #                       0.104,0.081,0.055,0.043,0.034])
        # true_grid = np.linspace(test_info['bin_ends'][0], test_info['bin_ends'][-1], len(true_amps) + 1)
        # true_grid_mids = (true_grid[1:] + true_grid[:-1]) / 2.
        # f = spi.interp1d(true_grid_mids, true_amps)
        # bin_mids = (test_info['bin_ends'][1:] + test_info['bin_ends'][:-1]) / 2.
        # bin_difs = test_info['bin_ends'][1:] - test_info['bin_ends'][:-1]
        # true_means = bin_mids
        # true_amps = f(bin_mids)
        # true_sigmas = bin_difs
    else:
        bin_range = max(test_info['bin_ends']) - min(test_info['bin_ends'])
        if test_info['params']['delta_truth'] == 1:
            true_amps = np.array([1.])
            true_means = np.array([1.]) + min(test_info['bin_ends'])
            true_sigmas = np.array([0.003])
        else:
            true_amps = np.array([0.20, 0.35, 0.55])
            true_means = np.array([0.5, 0.2, 0.75]) * bin_range + min(test_info['bin_ends'])
            true_sigmas = np.array([0.4, 0.2, 0.1]) * bin_range
        n_mix_comps = len(true_amps)
        true_funcs = []
        for c in range(n_mix_comps):
            true_funcs.append(chippr.gauss(true_means[c], true_sigmas[c]**2))
            true_nz = chippr.gmix(true_amps, true_funcs, limits=(min(test_info['bin_ends']), max(test_info['bin_ends'])))

            true_dict = {'amps': true_amps, 'means': true_means, 'sigmas': true_sigmas}
    true_dict['bins'] = test_info['bin_ends']

    # true_zs = true_nz.sample(test_info['params']['n_galaxies'])
    # true_dict['zs'] = true_zs
    test_info['truth'] = true_dict

    return(test_info, true_nz)

def make_interim_prior(given_key, wrong=False, grid=None):
    """
    Function to make the histogram-parametrized interim prior

    Parameters
    ----------
    given_key: string
        name of test case for which interim prior is to be made
    wrong: Boolean, optional
        wrongify the interim prior?
    grid: numpy.ndarray, float, optional
        grid for interim prior

    Returns
    -------
    interim_prior: chippr.discrete or chippr.gauss or chippr.gmix object
        the discrete distribution that will be the interim prior
    """
    test_info = all_tests[given_key]
    bin_mids = (test_info['bin_ends'][1:] + test_info['bin_ends'][:-1]) / 2.
    bin_difs = test_info['bin_ends'][1:] - test_info['bin_ends'][:-1]

    if not wrong:
        int_pr_type = test_info['params']['interim_prior']
        print('using true interim prior of '+int_pr_type)
    else:
        int_pr_type = test_info['params']['bias_model']
        print('using wrong interim prior of '+int_pr_type)

    if int_pr_type == 'template':
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
    elif int_pr_type == 'training':
        int_amps = np.array([0.150,0.822,1.837,2.815,3.909,
                              5.116,6.065,6.477,6.834,7.304,
                              7.068,6.771,6.587,6.089,5.165,
                              4.729,4.228,3.664,3.078,2.604,
                              2.130,1.683,1.348,0.977,0.703,
                              0.521,0.339,0.283,0.187,0.141,
                              0.104,0.081,0.055,0.043,0.034])
        int_grid = np.linspace(0., 1.1, len(int_amps) + 1)
        int_mids = (int_grid[1:] + int_grid[:-1]) / 2.
        int_amps = spi.griddata(int_mids, int_amps, bin_mids, method='linear', fill_value=np.min(int_amps), rescale=False)
        # # int_grid = np.concatenate((int_grid, np.array([test_info['bin_ends'][-1]])))
        # # print(int_grid)
        # int_amps.append(int_amps[-1])
        # int_amps = np.array(int_amps)
        # # this basically hardcodes in that grids start at 0.!
        #
        # int_grid_mids = (int_grid[1:] + int_grid[:-1]) / 2.
        #
        # int_grid_mids = np.concatenate((int_grid_mids, np.array([test_info['bin_ends'][-1]])))
        # f = spi.interp1d(int_grid_mids, int_amps)
        # int_amps = f(bin_mids)
        int_means = bin_mids
        int_sigmas = bin_difs
        n_mix_comps = len(int_amps)
        int_funcs = []
        for c in range(n_mix_comps):
            int_funcs.append(chippr.gauss(int_means[c], int_sigmas[c]**2))
        interim_prior = chippr.gmix(int_amps, int_funcs,
                limits=(min(test_info['bin_ends']), max(test_info['bin_ends'])))
    else:
        bin_ends = test_info['bin_ends']
        weights = np.ones_like(bin_mids)
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
                                test_info['params']['n_bins']+1)
    # print('bin ends to simulation = '+str(test_info['bin_ends']))

    test_info, true_nz = make_true(given_key)
    # true_amps = test_info['truth']['amps']
    # true_means = test_info['truth']['means']
    # true_sigmas =  test_info['truth']['sigmas']
    # n_mix_comps = len(true_amps)
    # true_funcs = []
    # for c in range(n_mix_comps):
    #     true_funcs.append(chippr.gauss(true_means[c], true_sigmas[c]**2))
    # true_nz = chippr.gmix(true_amps, true_funcs,
    #         limits=(test_info['params']['bin_min'], test_info['params']['bin_max']))

    interim_prior = make_interim_prior(given_key)
    print('providing interim prior '+str(interim_prior))

    posteriors = chippr.catalog(param_file_name, loc=test_dir, prepend=test_name)
    output = posteriors.create(true_nz, interim_prior, N=test_info['params']['n_gals'])
    print('using implicit prior '+str(posteriors.cat['log_interim_prior']))
    # data = np.exp(output['log_interim_posteriors'])
    # print('bin ends from simulation: '+str(posteriors.bin_ends))

    if test_info['params']['wrong_prior']:
        int_pr = make_interim_prior(given_key, wrong=True)
        int_pr_fine = np.array([int_pr.pdf(posteriors.z_fine)])
        int_pr_coarse = posteriors.coarsify(int_pr_fine)
        posteriors.cat['log_interim_prior'] = u.safe_log(int_pr_coarse[0])
    print('writing implicit prior '+str(posteriors.cat['log_interim_prior']))

    posteriors.write()
    data_dir = posteriors.data_dir
    with open(os.path.join(data_dir, 'true_params.p'), 'w') as true_file:
        pickle.dump(test_info['truth'], true_file)

if __name__ == "__main__":

    import numpy as np
    import scipy.stats as sps
    import pickle
    import shutil
    import os
    import scipy.interpolate as spi
    import timeit

    import chippr
    from chippr import *

    result_dir = os.path.join('..', 'results')
    test_name = 'single_lsst_trpr'
    all_tests = {}
    test_info = {}
    test_info['name'] = test_name
    all_tests[test_name] = test_info

    make_catalog(test_name)
