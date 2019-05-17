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
    b = 3.#1. / z_difs ** 2
    c = 3.e-2#a / n_pdfs
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

    Notes
    -----
    TODO: enable continuation of sampling starting from samples in a file
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
    simulated_posteriors = catalog(params=param_file_name, loc=test_dir, prepend=test_name)
    saved_location = 'data'
    saved_type = '.txt'
    data = simulated_posteriors.read(loc=saved_location, style=saved_type)
    zs = data['bin_ends']
    z_difs = zs[1:]-zs[:-1]
    with open(os.path.join(os.path.join(test_dir, saved_location), 'true_vals.txt'), 'r') as true_file:
        true_data = csv.reader(true_file, delimiter=' ')
        true_vals = []
        for z in true_data:
            true_vals.append(float(z[0]))
        true_vals = np.array(true_vals)
        true_vals = np.histogram(true_vals, bins=zs, density=True)[0]
    true_nz = chippr.discrete(zs, true_vals)

    (prior, cov) = set_up_prior(data, params)

    nz = log_z_dens(data, prior, truth=true_nz, loc=test_dir, prepend=test_name, vb=True)

    nz_stacked = nz.calculate_stacked()
    # print('stacked: '+str(np.dot(np.exp(nz_stacked), z_difs)))
    nz_mmap = nz.calculate_mmap()
    # print('MMAP: '+str(np.dot(np.exp(nz_mmap), z_difs)))
    # nz_mexp = nz.calculate_mexp()
    # print('MExp: '+str(np.dot(np.exp(nz_mexp), z_difs)))

    start_mmle = timeit.default_timer()
    nz_mmle = nz.calculate_mmle(nz_stacked, no_data=params['no_data'], no_prior=params['no_prior'])
    end_mmle = timeit.default_timer()-start_mmle
    print('MMLE: '+str(np.dot(np.exp(nz_mmle), z_difs))+' in '+str(end_mmle))

    nz_stats = nz.compare()
    nz.plot_estimators(log=True, mini=False)
    nz.plot_estimators(log=False, mini=False)
    nz.write('nz.p')

    # COMMENT OUT TO AVOID SAMPLING
    #start_mean = mvn(nz_mmle, cov).sample_one()
    start = prior#mvn(data['log_interim_prior'], cov)

    n_bins = len(zs) - 1
    if params['n_walkers'] is not None:
        n_ivals = params['n_walkers']
    else:
        n_ivals = 10 * n_bins
    initial_values = start.sample(n_ivals)

    start_samps = timeit.default_timer()
    nz_samps = nz.calculate_samples(initial_values, no_data=params['no_data'], no_prior=params['no_prior'], n_procs=1, gr_threshold=params['gr_threshold'])
    time_samps = timeit.default_timer()-start_samps
    print(test_name+' sampled '+str(params['n_accepted'])+' after '+str(nz.burn_ins * params['n_burned'])+' in '+str(time_samps))

    nz_stats = nz.compare()
    nz.plot_estimators(log=True, mini=False)
    nz.plot_estimators(log=False, mini=False)
    nz.write('nz.p')

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
