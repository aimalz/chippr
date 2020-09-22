# Wrapper for pomegranate.GeneralMixtureModel of pomegranate.distributions.UniformDistribution objects

import bisect

import numpy as np

from pomegranate import GeneralMixtureModel as GMM
from pomegranate.distributions import UniformDistribution as UD

import chippr
from chippr import defaults as d
from chippr import sim_utils as su

class discrete(object):
    def __init__(self, bin_ends, weights):
        """
        Binned function class for any discrete function

        Parameters
        ----------
        bin_ends: numpy.ndarray, float
            endpoints of bins
        weights: numpy.ndarray, float
            relative weights associated with each bin

        Notes
        -----
        TO DO: Rename to piecewise constant or somesuch
        """
        self.bin_ends = bin_ends
        self.dbins = self.bin_ends[1:] - self.bin_ends[:-1]
        self.n_bins = len(self.bin_ends)-1
        self.bin_range = list(range(self.n_bins))

        self.weights = weights
        # print('dbins: '+str(self.dbins))
        # print('weights: '+str((self.weights)))
        # print('sumweights: '+str((np.sum(self.weights))))
        # print('dotweights: '+str((np.dot(self.weights, self.dbins))))
        self.normweights = np.cumsum(self.weights) / np.sum(self.weights)
        # print('normweights: '+str(self.normweights))
        self.distweights = np.cumsum(self.weights) / np.dot(self.weights, self.dbins)
        # print('distweights: '+str(self.distweights))

        self.funcs = [UD(self.bin_ends[i], self.bin_ends[i+1]) for i in self.bin_range]
        if self.n_bins > 1:
            self.dist = GMM(self.funcs, weights=self.weights)
        else:
            self.dist = self.funcs[0]

    def pdf(self, xs):
        return self.evaluate(xs)

    def evaluate_one(self, x):
        """
        Function to evaluate the discrete probability distribution at one point

        Parameters
        ----------
        x: float
            value at which to evaluate discrete probability distribution

        Returns
        -------
        p: float
            value of discrete probability distribution at x
        """
        p = self.dist.probability(x)
        return p

    def evaluate(self, xs):
        """
        Function to evaluate the discrete probability distribution at many points

        Parameters
        ----------
        xs: ndarray, float
            values at which to evaluate discrete probability distribution

        Returns
        -------
        ps: ndarray, float
            values of discrete probability distribution at xs
        """
        # ps = np.array([self.evaluate_one(x) for x in xs])
        ps = self.dist.probability(xs)
        return ps

    def sample_one(self):
        """
        Function to sample a single value from discrete probability distribution

        Returns
        -------
        x: float
            a single point sampled from the discrete probability distribution
        """
        # r = np.random.random()
        # k = bisect.bisect(self.normweights, r)
        #
        # x = np.random.uniform(low=self.bin_ends[k], high=self.bin_ends[k+1])
        x = self.dist.sample(1)
        return x

    def sample(self, n_samps):
        """
        Function to take samples from discrete probability distribution

        Parameters
        ----------
        n_samps: int
            number of samples to take

        Returns
        -------
        xs: ndarray, float
            array of points sampled from the discrete probability distribution
        """
        # print('discrete trying to sample '+str(n_samps)+' from '+str(self.dist))
        # xs = np.array([self.sample_one() for n in range(n_samps)])
        xs = np.array(self.dist.sample(n_samps))
        # print('discrete sampled '+str(n_samps)+' from '+str(self.dist))
        return xs
