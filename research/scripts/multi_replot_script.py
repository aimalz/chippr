def set_up_prior(data):
    """
    Function to create prior distribution from data

    Parameters
    ----------
    data: dict
        catalog dictionary containing bin endpoints, log interim prior, and log interim posteriors

    Returns
    -------
    prior: chippr.mvn object
        prior distribution as multivariate normal
    """
    zs = data['bin_ends']
    log_nz_intp = np.exp(data['log_interim_prior'])
    log_z_posts = np.exp(data['log_interim_posteriors'])

    z_difs = zs[1:]-zs[:-1]
    z_mids = (zs[1:]+zs[:-1])/2.
    n_bins = len(z_mids)

    prior_var = np.eye(n_bins)
    for k in range(n_bins):
        prior_var[k] = 1. * np.exp(-0.5 * (z_mids[k] - z_mids) ** 2 / 0.05 ** 2)
    prior_mean = log_nz_intp
    prior = mvn(prior_mean, prior_var)
    return prior

def just_plot(given_key):
    """
    Function to make plot from a previously generate log_z_dens object

    Parameters
    ----------
    given_key: string
        name of test case to be run
    """
    test_info = all_tests[given_key]
    test_name = test_info['name']

    test_name = test_name[:-1]
    param_file_name = test_name + '.txt'

    test_dir = os.path.join(result_dir, test_name)
    simulated_posteriors = catalog(params=param_file_name, loc=test_dir)
    saved_location = 'data'
    saved_type = '.txt'
    data = simulated_posteriors.read(loc=saved_location, style=saved_type)

    # with open(os.path.join(os.path.join(test_dir, saved_location), 'true_params.p'), 'r') as true_file:
    #     true_nz_params = pickle.load(true_file)
    # true_amps = true_nz_params['amps']
    # true_means = true_nz_params['means']
    # true_sigmas =  true_nz_params['sigmas']
    # n_mix_comps = len(true_amps)
    # true_funcs = []
    # for c in range(n_mix_comps):
    #     true_funcs.append(chippr.gauss(true_means[c], true_sigmas[c]**2))
    # true_nz = chippr.gmix(true_amps, true_funcs,
    #         limits=(simulated_posteriors.params['bin_min'], simulated_posteriors.params['bin_max']))

    true_data = os.path.join(test_dir, saved_location)
    with open(os.path.join(true_data, 'true_vals.txt'), 'rb') as csvfile:
        tuples = (line.split(None) for line in csvfile)
        alldata = [[float(pair[k]) for k in range(0,len(pair))] for pair in tuples]
    true_vals = np.array(alldata).T
    bin_mids = (data['bin_ends'][1:] + data['bin_ends'][:-1]) / 2.
    catalog_plots.plot_obs_scatter(true_vals.T, np.exp(data['log_interim_posteriors']), bin_mids, plot_loc=os.path.join(test_dir, 'plots'))

    prior = set_up_prior(data)
    n_bins = len(data['log_interim_prior'])
    n_ivals = 2 * n_bins
    initial_values = prior.sample(n_ivals)

    true_nz = np.log(np.histogram(true_vals[0], bins=data['bin_ends'], density=True)[0])
    catalog_plots.plot_mega_scatter(true_vals.T, np.exp(data['log_interim_posteriors']), bin_mids, data['bin_ends'], truth=(bin_mids, np.exp(true_nz)), plot_loc=os.path.join(test_dir, 'plots'), prepend=test_name+'-', int_pr=(bin_mids, np.exp(data['log_interim_prior'])))

    # print(true_nz)
    true_nz = chippr.discrete(data['bin_ends'], true_nz)
    nz = log_z_dens(data, prior, truth=true_nz, loc=test_dir, prepend=test_name, vb=True)

    nz.info = nz.read('nz.p')
    # print(nz.info['stats'])
    nz_stats = nz.compare()

    prepend = test_name
    nz.add_text = prepend+'_'
    nz.plot_estimators(log=True, mini=False)
    nz.plot_estimators(log=False, mini=False)

    # nz.write('nz.p')

if __name__ == "__main__":

    import numpy as np
    import pickle
    import os
    import multiprocessing as mp

    import chippr
    from chippr import *

    result_dir = os.path.join('..', 'results')
    name_file = 'which_plt_tests.txt'

    with open(name_file) as tests_to_run:
        all_tests = {}
        for test_name in tests_to_run:
            test_info = {}
            test_info['name'] = test_name
            all_tests[test_name] = test_info

    nps = mp.cpu_count() - 1
    pool = mp.Pool(nps)
    pool.map(just_plot, all_tests.keys())
