def check_prob_params(params):
    """
    Sets parameter values pertaining to components of probability

    Parameters
    ----------
    params: dict
        dictionary containing key/value pairs for probability

    Returns
    -------
    params: dict
        dictionary containing key/value pairs for probability
    """
    if 'prior_mean' not in params:
        params['prior_mean'] = 'interim'
    else:
        params['prior_mean'] = params['prior_mean'][0]
    if 'no_prior' not in params:
        params['no_prior'] = 0
    else:
        params['no_prior'] = int(params['no_prior'][0])
    if 'no_data' not in params:
        params['no_data'] = 0
    else:
        params['no_data'] = int(params['no_data'][0])
    return params

# def make_true_nz(test_name):
#     """
#     Function to create true redshift distribution to be shared among several
#     test cases
#
#     Parameters
#     ----------
#     test_name: string
#         name used to look up parameters for making true_nz
#
#     Returns
#     -------
#     true_nz: chippr.gmix object
#         gaussian mixture probability distribution
#
#     Notes
#     -----
#     test_name is currently ignored but will soon be used to load parameters for making true_nz instead of hardcoded values.
#     """
#     true_amps = np.array([0.20, 0.35, 0.55])
#     true_means = np.array([0.5, 0.2, 0.75])
#     true_sigmas = np.array([0.4, 0.2, 0.1])
#
#     true_nz = chippr.gmix(true_amps, true_means, true_sigmas, limits=(0., 1.))
#
#     return true_nz

def set_up_prior(data, params):
    """
    Function to create prior distribution from data

    Parameters
    ----------
    data: dict
        catalog dictionary containing bin endpoints, log interim prior, and log
        interim posteriors
    params: dict
        dictionary of parameter values for creation of prior

    Returns
    -------
    prior: chippr.mvn object
        prior distribution as multivariate normal
    """
    zs = data['bin_ends']
    log_nz_intp = data['log_interim_prior']
    log_z_posts = data['log_interim_posteriors']

    z_difs = zs[1:]-zs[:-1]
    z_mids = (zs[1:]+zs[:-1])/2.
    n_bins = len(z_mids)

    n_pdfs = len(log_z_posts)

    a = 1.# / n_bins
    b = 20.#1. / z_difs ** 2
    c = a / n_pdfs
    prior_var = np.eye(n_bins)
    for k in range(n_bins):
        prior_var[k] = a * np.exp(-0.5 * b * (z_mids[k] - z_mids) ** 2)
    prior_var += c * np.identity(n_bins)

    prior_mean = log_nz_intp
    prior = mvn(prior_mean, prior_var)
    if params['prior_mean'] == 'sample':
        new_mean = prior.sample_one()
        prior = mvn(new_mean, prior_var)
        print(params['prior_mean'], prior_mean, new_mean)
    else:
        print(params['prior_mean'], prior_mean)

    return (prior, prior_var)

def do_inference(given_key):
    """
    Function to do inference from a catalog of photo-z interim posteriors

    Parameters
    ----------
    given_key: string
        name of test case to be run
    """
    test_info = all_tests[given_key]
    test_name = test_info['name']

    test_name = test_name[:-1]
    param_file_name = test_name + '.txt'

    params = chippr.utils.ingest(param_file_name)
    params = check_prob_params(params)
    params = defaults.check_inf_params(params)
    print(params)

    test_dir = os.path.join(result_dir, test_name)
    simulated_posteriors = catalog(params=param_file_name, loc=test_dir)
    saved_location = 'data'
    saved_type = '.txt'
    data = simulated_posteriors.read(loc=saved_location, style=saved_type)
    zs = data['bin_ends']
    z_difs = zs[1:]-zs[:-1]
    with open(os.path.join(os.path.join(test_dir, saved_location), 'true_params.p'), 'r') as true_file:
        true_nz_params = pickle.load(true_file)
    if true_nz_params['format'] == 'discrete':
        fine_grid = true_nz_params['grid']
        amps = true_nz_params['amps']
        true_funcs = [chippr.discrete(fine_grid, amps)]
        true_amps = [1.]
    elif true_nz_params['format'] == 'gauss':
        true_amps = true_nz_params['amps']
        true_means = true_nz_params['means']
        true_sigmas =  true_nz_params['sigmas']
        n_mix_comps = len(true_amps)
        true_funcs = []
        for c in range(n_mix_comps):
            true_funcs.append(chippr.gauss(true_means[c], true_sigmas[c]**2))
    true_nz = chippr.gmix(true_amps, true_funcs,
            limits=(min(zs), max(zs)))

    (prior, cov) = set_up_prior(data, params)

    nz = log_z_dens(data, prior, truth=true_nz, loc=test_dir, vb=True)

    nz_stacked = nz.calculate_stacked()
    print('stacked: '+str(np.dot(np.exp(nz_stacked), z_difs)))
    nz_mmap = nz.calculate_mmap()
    print('MMAP: '+str(np.dot(np.exp(nz_mmap), z_difs)))
    nz_mexp = nz.calculate_mexp()
    print('MExp: '+str(np.dot(np.exp(nz_mexp), z_difs)))
    nz_mmle = nz.calculate_mmle(nz_stacked, no_data=params['no_data'], no_prior=params['no_prior'])
    print('MMLE: '+str(np.dot(np.exp(nz_mmle), z_difs)))

    nz_stats = nz.compare()
    nz.plot_estimators()
    nz.write('nz.p')

    # # n_bins = len(nz_mmle)
    # if params['n_walkers'] is not None:
    #     n_ivals = params['n_walkers']
    # else:
    #     n_ivals = 10 * n_bins
    # initial_values = prior.sample(n_ivals)
    # log_z_dens_plots.plot_ivals(initial_values, nz.info, nz.plot_dir)
    # nz_samps = nz.calculate_samples(initial_values, no_data=params['no_data'], no_prior=params['no_prior'])
    #
    # nz_stats = nz.compare()
    #
    # nz.plot_estimators()
    # nz.write('nz.p')

if __name__ == "__main__":

    import numpy as np
    import pickle
    import os
    import multiprocessing as mp

    import chippr
    from chippr import *

    result_dir = os.path.join('..', 'results')
    name_file = 'which_inf_tests.txt'

    with open(name_file) as tests_to_run:
        all_tests = {}
        for test_name in tests_to_run:
            test_info = {}
            test_info['name'] = test_name
            all_tests[test_name] = test_info

    nps = mp.cpu_count()
    pool = mp.Pool(nps)
    pool.map(do_inference, all_tests.keys())
