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

# call the R code that I have written
robjects.r('source("ks_KDE.r")')


# --------- methods for computing gal-DM offsets-------------------------
def find_peaks_from_py_diff(fhat, estKey="estimate", gridKey="eval_points"):
    """
    :note: side effects
    """

    est = fhat[estKey]
    # diff consecutive columns
    colGrad1 = np.diff(est)
    colGrad1 = np.column_stack((colGrad1, np.zeros(est.shape[0])))

    # diff the consecutive cols in the reverse direction
    # or actually can made use of np.diff(est), multiply by -1 then add
    # zero column on the leftmost side of the arrays
    colGrad2 = np.diff(est.transpose()[::-1].transpose())
    colGrad2 = colGrad2.transpose()[::-1].transpose()
    colGrad2 = np.column_stack((np.zeros(est.shape[0]), colGrad2))

    colMask1 = np.logical_and(colGrad1 < 0, colGrad2 < 0)

    # diff consecutive rows
    rowGrad1 = np.diff(fhat[estKey], axis=0)
    rowGrad1 = np.vstack((rowGrad1, np.zeros(est.shape[0]).transpose()))

    # diff the consecutive row in the reverse direction
    rowGrad2 = np.diff(est[::-1], axis=0)
    rowGrad2 = np.vstack((rowGrad2, np.zeros(est.shape[0]).transpose()))
    rowGrad2 = rowGrad2[::-1]

    rowMask1 = np.logical_and(rowGrad1 < 0, rowGrad2 < 0)

    mask = np.logical_and(colMask1, rowMask1)
    rowIx, colIx = np.where(mask)

    rowIx, colIx = check_peak_higher_than_corner_values(fhat, rowIx, colIx)
    rowIx, colIx = sort_peaks_with_decreasing_density(fhat, rowIx, colIx)

    fhat["peaks_xcoords"] = fhat["eval_points"][0][rowIx]
    fhat["peaks_ycoords"] = fhat["eval_points"][1][colIx]
    fhat["peaks_rowIx"] = rowIx
    fhat["peaks_colIx"] = colIx

    return


def check_peak_higher_than_corner_values(fhat, rowIx, colIx,
                                         estKey="estimate",
                                         gridKey="eval_points",
                                         debug=False):
    """due to kludgy implementation I didn't check for corner values
    in the derivative function

    :param fhat: dictionary
    :param rowIX: list of integer
        row indices of the peak
    :param colIX: list of integer
        col indices of the peak
    """

    OK_peaks = np.array([check_corners_of_one_peak(fhat, rowIx[i], colIx[i])
                         for i in range(len(rowIx))], dtype=bool)
    if debug:
        print "OK_peaks = ", OK_peaks

    return rowIx[OK_peaks], colIx[OK_peaks]


def check_corners_of_one_peak(fhat, peakRowIx, peakColIx, debug=False):
    check_rowIx, check_colIx = check_ix(fhat, peakRowIx, peakColIx)

    if debug:
        print "peakRowIX, peakColIx = {0}, {1}".format(peakRowIx, peakColIx)
        print "checkRowIx = {0}".format(check_rowIx)
        print "checkColIx = {0}".format(check_colIx)

    OK = np.sum([fhat["estimate"][peakRowIx, peakColIx] >
                 fhat["estimate"][check_rowIx[i], check_colIx[i]]
                 for i in range(len(check_rowIx))]) == len(check_rowIx)

    if debug:
        print "OK or not = ", OK
    return OK


def check_ix(fhat, rowIx, colIx):
    """ compute ix of the corner values to be checked
    :param fhat: dictionary
    :param rowIX: integer
        row index of the peak
    :param colIX: integer
        col index of the peak
    """
    check_rowIx = []
    check_colIx = []
    upper_rowIx = fhat["eval_points"][0].shape
    upper_colIx = fhat["eval_points"][1].shape

    # upper left corner
    if rowIx > 0 and colIx > 0:
        check_rowIx.append(rowIx - 1)
        check_colIx.append(colIx - 1)

    # upper right corner
    if rowIx > 0 and colIx < upper_colIx:
        check_rowIx.append(rowIx - 1)
        check_colIx.append(colIx + 1)

    # lower left corner
    if rowIx < upper_rowIx and colIx > 0:
        check_rowIx.append(rowIx + 1)
        check_colIx.append(colIx - 1)

    # lower right corner
    if rowIx < upper_rowIx and colIx < upper_colIx:
        check_rowIx.append(rowIx + 1)
        check_colIx.append(colIx + 1)

    return check_rowIx, check_colIx


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


def compute_KDE_peak_offsets(df, f, clstNo, cut_method, cut_kwargs, w=None,
                             verbose=False):
    """
    :params df: pandas dataframe for each cluster
    :params cut_method: function
    :params w: floats, weight

    :return: list of [offset, offsetR200]
        offset: offset in unit of c kpc/h
        offset: offset in terms of the R200C of the cluster
    :to do:
        want to make col a param
    :note:
        can think of making this function even more general
        by having the peak inference function passed in
    """
    # prepare the data for KDE
    mask = cut_method(df, **cut_kwargs)
    if verbose:
        print "# of subhalos after the cut = {0}".format(np.sum(mask))

    col = ["SubhaloPos0", "SubhaloPos1"]
    data = np.array(df[col][mask])
    print "data shape is ", data.shape

    results = do_KDE_and_get_peaks(data)
    # peaks = results[0]  # the first component give an R matrix of the peaks
    # peaks = np.array(peaks)[0]  # get only the first peak

    R200C = f["Group"]["Group_R_Crit200"][clstNo]

    fhat = convert_fhat_to_dict(results)  # [1])
    find_peaks_from_py_diff(fhat, estKey="estimate", gridKey="eval_points")
    fhat["peaks_dens"] = get_density_weights(fhat)

    # we have sorted the density so that the highest density peak is the first
    peaks = np.array(fhat["peaks_xcoords"][0], fhat["peaks_ycoords"][0])
    offset = np.sqrt(np.dot(peaks, peaks))
    offsetR200 = offset / R200C

    return [offset, offsetR200, fhat]

# ------------python wrapper to ks_KDE.r code ---------------------------


def convert_fhat_to_dict(r_fhat):
    """preserves the returned object structure with a dict
    :param r_fhat: robject of the output evaluated from ks.KDE

    :stability: works but may not be correct
    The R object has been modified

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


def get_density_weights(fhat, ix_rkey="peaks_rowIx",
                        ix_ckey="peaks_colIx",
                        pt_key="eval_points"):
    """
    :note: fhat is passed by reference, fhat is modified!
    """
    rowIx = fhat[ix_rkey]
    colIx = fhat[ix_ckey]
    peak_dens = np.array(fhat["estimate"][rowIx, colIx])

    return peak_dens / np.max(peak_dens)  # give relative weights


def py_2D_arr_to_R_matrix(x):
    """flattens the array, convert to R float vector then to R matrix
    x = np.array, with shape (dataNo, 2)
    """
    nrow = x.shape[0]
    x = robjects.FloatVector(np.concatenate([x[:, 0], x[:, 1]]))
    return robjects.r['matrix'](x, nrow=nrow)


def gaussian_mixture_data(samp_no=int(5e2), cwt=1. / 11., set_seed=True):
    """ thin wrapper around R function
    :params samp_no: integer,
        how many data points to be drawn
    :params cwt: float,
        weight for the central gaussian mixture out of the 3 mixtures
    :params set_seed: bool
        whether to set the seed or not

    :returns: R matrix of coords
    """
    return robjects.r["gaussian_mixture_data"](samp_no, cwt, set_seed=True)


def get_peaks(fhat, no_of_peaks=1):
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


# def find_peaks_from_2nd_deriv(fhat, verbose=False):
#     """not tested but works without errors
#     fhat = robject returned by ks.KDE
#     """
#     func = robjects.r["find_peaks_from_2nd_deriv"]
#
#     return func(fhat, verbose)


def do_KDE_and_get_peaks(x, w=None, dom_peak_no=1):
    """ don't want to write this for a general bandwidth selector yet
    :params x: np.array, each row should be one observation / subhalo
    :params w: np.float, weight of each row of data

    :returns list of 2 R objects:
        :R matrix of peaks: each row correspond to coordinates of one peak
        :R object: fhat this should be fed to convert_fhat_to_dict()
            if you wish to examine the object in python

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


# def do_KDE(data, bw_selector="Hscv", w=None, verbose=False):
#     """
#     :param data: np.array, with shape (dataNo, 2)
#     :param bw_selector: str, either 'Hscv' or 'Hpi'
#     :param w: np.array of floats that denote the weight for each data point
#     :param verbose: bool
#
#     :return: fhat, ks R object spat out by KDE()
#     """
#     assert data.shape[1] == 2, \
# "data array is of the wrong shape, want array with shape (# of obs, 2)"
#     assert bw_selector == 'Hscv' or bw_selector == 'Hpi', \
#         "bandwidth selector {0} not available".format(bw_selector)
#
#     data = py_2D_arr_to_R_matrix(data)
#     doKDE = robjects.r["do_KDE"]
#
#     if w is not None:
# needs to be tested
#         fhat = doKDE(data, robjects.r[bw_selector],
#                      robjects.FloatVector(w))
#     else:
#         fhat = doKDE(data, robjects.r[bw_selector])
#
#     return fhat

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



def sort_peaks_with_decreasing_density(fhat, rowIx, colIx):
    order = np.argsort(fhat["estimate"][rowIx, colIx])[::-1]
    sortedRowIx = np.array([rowIx[i] for i in order])
    sortedColIx = np.array([colIx[i] for i in order])

    return sortedRowIx, sortedColIx


# def convert_R_peak_ix_to_py_peaks(fhat, ix_key="peak_coords_ix",
#                                   pt_key="eval_points"):
#     """
#     :param fhat: python dictionary
#     :returns coords: numpy array
#         shape=(n_peak, 2)
#
#     :stability: this should be tested!
#     """
#     py_peak_ix = fhat[ix_key] - 1  # python is zeroth indexed
#     return np.array([[fhat[pt_key][0, ix[0]], fhat[pt_key][1, ix[1]]]
#                      for ix in py_peak_ix])
