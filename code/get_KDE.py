"""
Provides python wrapper around the R ks package for the KDE functions
I try to keep a one-to-one correspondane between the R functions and the
Python functions

Prerequisite:
* ks R package and its dependencies should be installed.
* ks_KDE.r (or a soft link) should be in the same directory for which you import / use get_KDE.py

Author: Karen Ng <karenyng@ucdavis.edu>
License: BSD
"""
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
base = importr("base")  # not really needed in the script

# call the R code that I have written
robjects.r('source("ks_KDE.r")')


# --------- functions for computing gal-DM offsets-------------------------
def find_peaks_from_py_diff(fhat, estKey="estimate", gridKey="eval_points"):
    """
    :param fhat: python dictionary
        :key estKey: value corresponds to 2D numpy array
            gives density estimate
        :key gridKey: value corresponds to 2D numpy array
            each array contains spatial coordinate in the following form
            i.e. [[x1, x2, ..., xn], [y1, y2, ..., yn]]
    :note: side effects for fhat, fhat is passed by reference
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
    rowGrad1 = np.vstack((rowGrad1, np.zeros(est.shape[1]).transpose()))

    # diff the consecutive row in the reverse direction
    rowGrad2 = np.diff(est[::-1], axis=0)
    rowGrad2 = np.vstack((rowGrad2, np.zeros(est.shape[1]).transpose()))
    rowGrad2 = rowGrad2[::-1]

    rowMask1 = np.logical_and(rowGrad1 < 0, rowGrad2 < 0)

    mask = np.logical_and(colMask1, rowMask1)
    rowIx, colIx = np.where(mask)

    rowIx, colIx = check_peak_higher_than_corner_values(fhat, rowIx, colIx)
    rowIx, colIx = sort_peaks_with_decreasing_density(fhat, rowIx, colIx)

    fhat["peaks_xcoords"] = fhat[gridKey][0][rowIx]
    fhat["peaks_ycoords"] = fhat[gridKey][1][colIx]
    fhat["peaks_rowIx"] = rowIx
    fhat["peaks_colIx"] = colIx

    return


def check_peak_higher_than_corner_values(fhat, rowIx, colIx,
                                         estKey="estimate",
                                         gridKey="eval_points",
                                         debug=False):
    """this function corrects for the fact that I did not
    check for corner values in `find_peaks_from_py_diff`

    :param fhat: dictionary
    :param rowIX: list of integer
        row indices of the peak
    :param colIX: list of integer
        col indices of the peak

    :return: two vectors of integers,
        rowIx contains row index (ix) of checked peaks
        colIx contains col index (ix) of checked peaks
    """

    OK_peaks = np.array([check_corners_of_one_peak(fhat, rowIx[i], colIx[i])
                         for i in range(len(rowIx))], dtype=bool)
    if debug:
        print "OK_peaks = ", OK_peaks

    return rowIx[OK_peaks], colIx[OK_peaks]


def check_corners_of_one_peak(fhat, peakRowIx, peakColIx, debug=False):
    """
    :param fhat: dictionary
    :param peakRowIx: list of integer
        row indices (ix) of the peak
    :param peakColIx: list of integer
        col indices (ix) of the peak

    :return: boolean
    """
    # return the indices of corner pixels to be compared against
    check_rowIx, check_colIx = check_ix(fhat, peakRowIx, peakColIx)

    if debug:
        print "peakRowIX, peakColIx = {0}, {1}".format(peakRowIx, peakColIx)
        print "checkRowIx = {0}".format(check_rowIx)
        print "checkColIx = {0}".format(check_colIx)

    # check peak density estimate > corner density estimate
    # for ALL 4 of the corner pixels
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


# ------------python wrapper to ks_KDE.r code ---------------------------

def convert_rfhat_to_dict(r_fhat):
    """preserves the returned object structure with a dict
    :param r_fhat: robject of the output evaluated from ks.KDE

    under this conversion

    Returns
    ======
    python dictionary

    fhat["eval_points"] : np.array of 2 arrays
        1st array is the 1st-coordinates of the grid
        2nd array is the 2nd-coordinates of the grid

    fhat["estimate"] : np.array
        density estimate at the grid location specified
        by `fhat["eval_points"]`

    fhat["bandwidth_matrix_H"] : np.array
        covariance matrix of the Gaussian kernel that is used
        you can imagine this to specify the best kernel smoothing width

    :to do: convert this to a h5 object instead
    """
    return {"eval_points": np.array(r_fhat[1]),  # fixed 2D size
            "estimate": np.array(r_fhat[2]),  # fixed size
            "bandwidth_matrix_H": np.array(r_fhat[3]),  # fixed size
            # "gridtype": tuple(r_fhat[4]),   # ('linear', 'linear')
            # "gridded": bool(r_fhat[5]),  # don't really have to store this
            # "binned": bool(r_fhat[6]),  # don't really have to store this
            # "names": list(r_fhat[7]),  # useless
            # "weight_w": np.array(r_fhat[8]),  # don't have to store
            # "data_x": np.array(r_fhat[0]),  # don't have to store
            }


def get_density_weights(fhat, ix_rkey="peaks_rowIx",
                        ix_ckey="peaks_colIx",
                        pt_key="eval_points"):
    """
    :param fhat: python dict containing the following keys
    :param ix_rkey: string, key of dict / df that contains the rowIx
    :param ix_ckey: string, key of dict / df that contains the colIx
    :param pt_ckey: string, key of dict / df that contains the eval_point
    :note: fhat is passed by reference, fhat is modified!
    """
    rowIx = fhat[ix_rkey]
    colIx = fhat[ix_ckey]
    peak_dens = np.array(fhat["estimate"][rowIx, colIx])

    # give relative weights
    fhat["max_peak_dens"] = np.max(peak_dens)
    fhat["peaks_dens"] = peak_dens / fhat["max_peak_dens"]
    return


def py_2D_arr_to_R_matrix(x):
    """flattens the array, convert to R float vector then to R matrix
    x = np.array, with shape (dataNo, 2)

    TODO: think of a more general way of converting between python and R data
    structures
    """
    nrow = x.shape[0]
    x = robjects.FloatVector(np.concatenate([x[:, 0], x[:, 1]]))
    return robjects.r['matrix'](x, nrow=nrow)


def py_1D_arr_to_R_vector(x):
    """flattens the array, convert to R float vector then to R matrix
    x = np.array, with shape (dataNo)

    TODO: think of a more general way of converting between python and R data
    structures
    """
    return robjects.FloatVector(x)


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
    return robjects.r["gaussian_mixture_data"](samp_no, cwt, set_seed=set_seed)


def do_1D_KDE(data, weight=None, convert_to_py_format=True):
    """
    :data: numpy 1D array
    :weight: TODO
    :convert_to_py_format: TODO
    :returns: TODO

    """

    if weight is None:
        weight = np.ones(len(data))

    do_KDE_in_R = robjects.r["do_KDE"]
    data = np.array(data)
    if data.ndim == 1:
        data = py_1D_arr_to_R_vector(data)
    else:
        raise ValueError(
            "Argument `data` needs to be a 1D array / list")


    weight = robjects.FloatVector(weight)
    r_fhat = do_KDE_in_R(data, w=weight)

    if not convert_to_py_format:
        return r_fhat
    else:
        return convert_rfhat_to_dict(r_fhat)


def do_KDE(x, w=None, dom_peak_no=1, convert_to_py_format=False):
    """
    :param x: np.array, each row should be one observation / subhalo
    :param w: np.float, weight of each row of data
    :param convert_to_py_format: bool, whether to return r object
        or python dictionary

    :returns list of 2 R objects:
        :R matrix of peaks: each row correspond to coordinates of one peak
        :R object: fhat this should be fed to convert_rfhat_to_dict()
            if you wish to examine the object in python
    or it returns a dictionary.
    See `convert_rfhat_to_dict` for more info about the returned dictionary

    """

    x = np.array(x)
    if w is None:
        w = np.ones(x.shape[0])
    w = robjects.FloatVector(w)
    x = py_2D_arr_to_R_matrix(x)

    do_KDE_in_R = robjects.r["do_KDE"]


    r_fhat = do_KDE_in_R(x, w=w, dom_peak_no=dom_peak_no)

    if not convert_to_py_format:
        return r_fhat
    else:
        return convert_rfhat_to_dict(r_fhat)


def do_KDE_and_get_peaks(x, w=None, dom_peak_no=1):
    """ do KDE and also put the peak location information into fhat
    :params x: np.array, each row should be one observation / subhalo
    :params w: np.float, weight of each row of data
    :dom_peak_no: (DEPRECIATED)

    :return: python dictionary - fhat
    :keys bandwidth_matrix_H: covariance matrix
    :keys peaks_dens: array of floats,
        relative density of the peaks arranged in same order
        as the coordinate arrays described below.
        The densities are computed relatively by dividing out the density of
        the densest peak.
    :keys peaks_ycoords: np array of 2nd coordinates of the found peaks
    :keys peaks_xcoords: np array of 1st coordinates of the found peaks
    :keys peaks_rowIx: np array
        of index of the 1st coordinates of the found peaks from
        fhat["estimate"] or fhat["eval_points"][0]
    :keys peaks_rowIx: np array
        of index of the 2nd coordinates of the found peaks from
        fhat["estimate"] or fhat["eval_points"][1]
    """
    res = do_KDE(x, w=w, dom_peak_no=dom_peak_no)
    fhat = convert_rfhat_to_dict(res)
    find_peaks_from_py_diff(fhat, estKey="estimate", gridKey="eval_points")
    get_density_weights(fhat)
    return fhat


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

    return do_KDE_and_get_peaks(fhat)


def sort_peaks_with_decreasing_density(fhat, rowIx, colIx):
    """
    :param fhat: dictionary
    :param rowIx: list of integers
    :param colIx: list of integers

    :return sortedRowIx: sorted list of integers
    :return sortedColIx: sorted list of integers
    """
    order = np.argsort(fhat["estimate"][rowIx, colIx])[::-1]
    sortedRowIx = np.array([rowIx[i] for i in order])
    sortedColIx = np.array([colIx[i] for i in order])

    return sortedRowIx, sortedColIx


def get_KDE_conf_reg(data_realizations, second_peak=False):
    """
    :param data_realizations: list of bootstrapped fhat data_realizations
    :param 2nd_peak: boolean, whether we want the result from the
        2nd peak

    :return fhat: dictionary of properties
        of KDE peaks
        from the data realizations and the peaks
    """
    KDE_fhat2 = [do_KDE_and_get_peaks(g_data)
                 for g_data in data_realizations]
    KDE_peaks2 = np.array([np.array([fhat2["peaks_xcoords"][0],
                                     fhat2["peaks_ycoords"][0]])
                           for fhat2 in KDE_fhat2])
    KDE_peak_dens2 = do_KDE(KDE_peaks2)
    KDE_peak_dens2 = convert_rfhat_to_dict(KDE_peak_dens2)
    find_peaks_from_py_diff(KDE_peak_dens2)

    # get second KDE peak
    if second_peak:
        KDE_peaks2b = np.array([np.array([fhat2["peaks_xcoords"][1],
                                         fhat2["peaks_ycoords"][1]])
                                for fhat2 in KDE_fhat2
                                if len(fhat2["peaks_xcoords"]) > 1])
        KDE_peak_dens2b = do_KDE(KDE_peaks2b)
        KDE_peak_dens2b = convert_rfhat_to_dict(KDE_peak_dens2b)
        find_peaks_from_py_diff(KDE_peak_dens2b)

        return KDE_peak_dens2, KDE_peak_dens2b

    return KDE_peak_dens2
