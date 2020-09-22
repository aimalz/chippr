# Module containing generally handy functions used by simulation and inference modules

import numpy as np

import chippr
from chippr import defaults as d

# Choose random seed at import, hopefully carries through everywhere
np.random.seed(d.seed)

def safe_log(arr, threshold=d.eps):
    """
    Takes the natural logarithm of an array that might contain zeros.

    Parameters
    ----------
    arr: ndarray, float
        array of values to be logged
    threshold: float, optional
        small, positive value to replace zeros and negative numbers

    Returns
    -------
    logged: ndarray
        logged values, with small value replacing un-loggable values
    """
    arr = np.asarray(arr)
    # if type(arr) == np.ndarray:
    arr[arr < threshold] = threshold
    # else:
    #     arr = max(threshold, arr)
    logged = np.log(arr)
    return logged

def ingest(in_info):
    """
    Function reading in parameter file to define functions necessary for
    generation of posterior probability distributions

    Parameters
    ----------
    in_info: string or dict
        string containing path to plaintext input file or dict containing
        likelihood input parameters

    Returns
    -------
    in_dict: dict
        dict containing keys and values necessary for posterior probability
        distributions
    """
    if type(in_info) == str:
        with open(in_info) as infile:
            lines = (line.split(None) for line in infile)
            in_dict = {defn[0] : defn[1:] for defn in lines}
    else:
        in_dict = in_info
    return in_dict

def mids_from_ends(inarr):
    """
    Function to make midpoints from grid of endpoints

    Parameters
    ----------
    inarr: array, shape=(N)
        array of grid endpoints

    Returns
    -------
    outarr: array, shape=(N-1)
        array of corresponding midpoints
    """
    outarr = (inarr[1:] + inarr[:-1]) / 2.
    return outarr

def ends_from_mids(inarr):
    """
    Function to make endpoints from grid of midpoints, assuming equal spacing

    Parameters
    ----------
    inarr: array, shape=(N)
        array of grid midpoints

    Returns
    -------
    outarr: array, shape=(N+1)
        array of corresponding endpoints
    """
    dif = np.mean(inarr[1:] - inarr[:-1])
    int_ends = mids_from_ends(inarr)
    outarr = np.concatenate((np.array([int_ends[0] - dif]), int_ends, np.array([int_ends[-1] + dif])))
    return outarr
