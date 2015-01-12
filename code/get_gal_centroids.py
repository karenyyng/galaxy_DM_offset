""" various functions for inferring centroids of galaxy population
Provides python wrapper around the R ks package for the KDE functions
I try to keep a one-to-one correspondane between the R functions and the
Python functions
"""
from __future__ import division
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
base = importr("base")  # not really needed in the script
# ks = importr("ks")

# call the R code that I have written
robjects.r('source("ks_KDE.r")')


# --------- methods for computing gal-DM offsets-------------------------
def cut_reliable_galaxies(df, DM_cut=1e3, star_cut=1e2):
    """ consider all cluster galaxies with minimal cuts
    :params df: pandas dataframe contains one cluster
    :params DM_cut: integer, how many DM particles needed for us to consider
        subhalos to be reliable
    :params star_cut: integer, how many stellar particles needed for us to
        consider subhalos to be reliable

    :notes:
    http://illustris-project.org/w/index.php/Data_Details#Snapshot_Contents
    """
    # DM cut
    mask = df["SubhaloLenType1"] > DM_cut
    return np.logical_and(mask, df["SubhaloLenType4"] > star_cut)


def compute_KDE_peak_offsets(df, f, clstNo, cut_method, cut_kwargs,
                             verbose=False):
    """
    :params df: pandas dataframe for each cluster
    :params cut_method: function

    :return: list of [offset, offsetR200]
        offset: offset in unit of c kpc/h
        offset: offset in terms of the R200C of the cluster

    :note:
        can think of making this function even more general
        by having the peak inference function passed in
    """
    # prepare the data for KDE
    mask = cut_method(df, **cut_kwargs)
    if verbose:
        print "# of subhalos after the cut = {0}".format(np.sum(mask))
    data = np.array([df.SubhaloPos0[mask],
                     df.SubhaloPos1[mask]]).transpose()

    peaks = do_KDE_and_get_peaks(data)
    peaks = np.array(peaks)[0]

    offset = np.sqrt(np.dot(peaks, peaks))
    R200C = f["Group"]["Group_R_Crit200"][clstNo]
    offsetR200 = offset / R200C

    return [offset, offsetR200]

# ------------python wrapper to ks_KDE.r code ---------------------------

def convert_fhat_to_dict(r_fhat):
    """preserves the returned object structure with a dict
    :param r_fhat: robject of the output evaluated from ks.KDE

    :stability: works but should be tested
    if I am not lazy I would write a proper class instead ;)
    can think about it if i have designated class methods for class vars

    under this conversion

    fhat["data_x"] : np.array with shape as (obs_no, 2)
    fhat["domPeaks"] : np.array with shape as (peak_no, 2)
    """

    return {"data_x": np.array(r_fhat[0]),
            "eval_points": np.array(r_fhat[1]),
            "estimate": np.array(r_fhat[2]),
            "bandwidth_matrix_H": np.array(r_fhat[3]),
            "gridtype": tuple(r_fhat[4]),
            "gridded": bool(r_fhat[5]),
            "binned": bool(r_fhat[6]),
            "names": list(r_fhat[7]),
            "weight_w": np.array(r_fhat[8])}


def py_2D_arr_to_R_matrix(x):
    """flattens the array, convert to R float vector then to R matrix
    x = np.array, with shape (dataNo, 2)
    """
    nrow = x.shape[0]
    x = robjects.FloatVector(np.concatenate([x[:, 0], x[:, 1]]))
    return robjects.r['matrix'](x, nrow=nrow)


def gaussian_mixture_data(samp_no=int(5e2), cwt=1. / 11.):
    return robjects.r["gaussian_mixture_data"](samp_no, cwt)


def do_KDE(data, bw_selector="Hscv", w=None, verbose=False):
    """
    :param data: np.array, with shape (dataNo, 2)
    :param bw_selector: str, either 'Hscv' or 'Hpi'
    :param w: np.array of floats that denote the weight for each data point
    :param verbose: bool

    :return: fhat, ks R object spat out by KDE()
    """
    assert data.shape[1] == 2, \
        "data array is of the wrong shape, want array with shape (# of obs, 2)"
    assert bw_selector == 'Hscv' or bw_selector == 'Hpi', \
        "bandwidth selector {0} not available".format(bw_selector)

    data = py_2D_arr_to_R_matrix(data)
    doKDE = robjects.r["do_KDE"]

    if w is not None:
        # needs to be tested
        fhat = doKDE(data, robjects.r[bw_selector],
                     robjects.FloatVector(w))
    else:
        fhat = doKDE(data, robjects.r[bw_selector])

    return fhat


def get_peaks(fhat, no_of_peaks):
    """
    :param fhat: robject spat out from ks.KDE
    :param no_of_peaks: integer

    :returns: list of peaks,
        each row correspond to one peak [[x1, y1], [x2, y2], ...]
    """
    findPeaks = robjects.r["find_peaks_from_2nd_deriv"]
    findDomPeaks = robjects.r["find_dominant_peaks"]

    peaks_ix = findPeaks(fhat)  # fhat[2] = fhat$estimate
    dom_peaks = np.array(findDomPeaks(fhat, peaks_ix, no_of_peaks))

    fhat = convert_fhat_to_dict(fhat)
    # subtracting 1 from the peak_coords since python is zeroth index,
    # R has 1 as the first index
    fhat["peaks_py_ix"] = np.array(peaks_ix) - 1

    # need to double check how the coordinates are put into vectors
    # something might have been transposed ...
    fhat["domPeaks"] = dom_peaks

    return fhat


def find_peaks_from_2nd_deriv(fhat, verbose=False):
    """not tested but works without errors
    fhat = robject returned by ks.KDE
    """
    func = robjects.r["find_peaks_from_2nd_deriv"]

    return func(fhat, verbose)


def do_KDE_and_get_peaks(x, w=None, dom_peak_no=1):
    """ don't want to write this for a general bandwidth selector yet

    :stability: untested
    """
    do_KDE_and_get_peaks = robjects.r["do_KDE_and_get_peaks"]

    x = py_2D_arr_to_R_matrix(np.array(x))
    if w is not None:
        w = robjects.FloatVector(w)
        res = do_KDE_and_get_peaks(x, w=w, dom_peak_no=dom_peak_no)
    else:
        res = do_KDE_and_get_peaks(x, dom_peak_no=dom_peak_no)

    return res


def bootstrap_KDE(data, bootNo=4, ncpus=2):
    """
    :params data: robject vector list ...
    :params bootNo: integer number of bootstrap samples to call

    :returns:
        list of peak values
    """
    func = robjects.r["bootstrap_KDE"]

    return func(data, bootNo=bootNo, ncpus=ncpus)


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
    # robjects.r[""]

    return None


# -----------other centroid methods ------------------------------------



def shrinking_apert(r0, x0, y0, data):
    """
    :param r0: float, aperture to consider in the data
    :param x0: float, initial x coord of center
    :param y0: float, initial y coord of center
    :param data: 2D np.array
    """
    return


def BCG():
    return


