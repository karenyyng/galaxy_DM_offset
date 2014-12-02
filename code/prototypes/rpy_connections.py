"""Python wrapper around the R ks package
Try to keep a one-to-one correspondane between the R functions and the
Python functions
"""
from __future__ import division
import numpy as np
from scipy.stats import multivariate_normal as mvnorm
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
ks = importr("ks")
base = importr("base")

# call the R code that I have written
robjects.r('source("ks_KDE.r")')


def convert_fhat_to_dict(r_fhat):
    """preserves the returned object structure with a dict
    :param: r_fhat = robject of the output evaluated from ks.KDE

    :stability: works but should be tested
    """

    return {"x": np.array(r_fhat[0]).transpose(),
            "eval.points": np.array(r_fhat[1]),
            "estimate": np.array(r_fhat[2]),
            "H": np.array(r_fhat[3]),
            "gridtype": tuple(r_fhat[4]),
            "gridded": bool(r_fhat[5]),
            "binned": bool(r_fhat[6]),
            "names": list(r_fhat[7]),
            "w": np.array(r_fhat[8])}


def do_KDE(data, bandwidth_selector, w=None):
    func = robjects.r["do_KDE"]

    if w is not None:
        return convert_fhat_to_dict(func(data, bandwidth_selector, w))
    else:
        return convert_fhat_to_dict(func(data, bandwidth_selector))


def TwoDtestCase1(samp_no=5e2, cwt=1. / 11., w=None, H=None):
    """call the TwoDtestCase1 for getting data with 3 Gaussian mixtures
    """
    func = robjects.r["TwoDtestCase1"]
    if w is not None:
        return convert_fhat_to_dict(func(samp_no, cwt, w))
    else:
        return convert_fhat_to_dict(func(samp_no, cwt))


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

    return



