""" various functions for inferring centroids of galaxy population
Provides python wrapper around the R ks package for the KDE functions
I try to keep a one-to-one correspondane between the R functions and the
Python functions
"""
from __future__ import division
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
ks = importr("ks")
base = importr("base")

# call the R code that I have written
robjects.r('source("ks_KDE.r")')


#------------python wrapper to ks_KDE.r code ---------------------------

def convert_fhat_to_dict(r_fhat):
    """preserves the returned object structure with a dict
    :param: r_fhat = robject of the output evaluated from ks.KDE

    :stability: works but should be tested
    if I am not lazy I would write a proper class instead ;)
    """

    return {"data_x": np.array(r_fhat[0]).transpose(),
            "eval_points": np.array(r_fhat[1]),
            "estimate": np.array(r_fhat[2]),
            "bandwidth_matrix_H": np.array(r_fhat[3]),
            "gridtype": tuple(r_fhat[4]),
            "gridded": bool(r_fhat[5]),
            "binned": bool(r_fhat[6]),
            "names": list(r_fhat[7]),
            "weight_w": np.array(r_fhat[8])}


def do_KDE(data, bandwidth_selector, w=None, verbose=False):
    doKDE = robjects.r["do_KDE"]

    if w is not None:
        fhat = doKDE(data, bandwidth_selector, w)
    else:
        fhat = doKDE(data, bandwidth_selector)

    return fhat


def get_peaks(fhat):
    """
    fhat = robject spat out from ks.KDE
    """
    findPeaks = robjects.r["find_peaks_from_2nd_deriv"]
    findDomPeaks = robjects.r["find_dominant_peaks"]

    peaks_ix = findPeaks(fhat)  # fhat[2] = fhat$estimate
    dom_peaks = np.array(findDomPeaks(fhat, peaks_ix))

    fhat = convert_fhat_to_dict(fhat)
    # subtracting 1 from the peak_coords since python is zeroth index, R is
    # not
    fhat["peaks_py_ix"] = np.array(peaks_ix) - 1
    fhat["domPeaks"] = dom_peaks

    return fhat


def TwoDtestCase1(samp_no=5e2, cwt=1. / 11., w=None, H=None):
    """call the TwoDtestCase1 for getting data with 3 Gaussian mixtures
    """
    func = robjects.r["TwoDtestCase1"]

    if w is not None:
        fhat = func(samp_no, cwt, w)
    else:
        fhat = func(samp_no, cwt)

    return get_peaks(fhat)


def rmvnorm_mixt(n, mus, Sigmas, props):
    """
    parse arguments as string to call functions that I wrote in R
    :params:
    n = integer, number of samples
    mus = numpy array,
    :return:
    x = robject, more specifically, R vector,
        that contains coordinates of the normal mixture
    """
    # need to import a library
    #robjects.r[""]

    return None


def find_peaks_from_2nd_deriv(dens, verbose=False):
    """untested"""
    func = robjects.r["find_peaks_from_2nd_deriv"]

    return func(dens, verbose)


def bootstrapped_KDE_peaks():
    return

#-----------other centroid methods ------------------------------------

def shrinking_apert():
    return


def BCG():
    return


