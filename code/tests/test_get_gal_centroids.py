"""unit tests for get gal centroids.py"""
from __future__ import (division, print_function, unicode_literals)
import sys
sys.path.append("../")
sys.path.append("../analyses/")
from get_gal_centroids import *
from plot_gal_prop import *
from compare_peak_methods import draw_gaussian
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def test_data1():
    fhat = {}
    fhat["estimate"] = np.array([[1, 1, 1, 1, 1, 1, 1],
                                 [1, 8, 3, 2, 1, 1, 1],
                                 [1, 7, 9, 3, 1, 1, 1],
                                 [3, 2, 4, 4, 5, 2, 1],
                                 [1, 1, 3, 2, 1, 1, 1],
                                 [1, 1, 1, 1, 2, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1]])

    fhat["eval_points"] = np.array([range(fhat["estimate"].shape[0]),
                                    range(fhat["estimate"].shape[1])])

    return fhat


def test_find_peaks_from_py_deriv(test_data1):
    fhat = test_data1
    # ans put in descending density
    correct_peaks_rowIx = [2, 3]
    correct_peaks_colIx = [2, 4]

    find_peaks_from_py_diff(fhat)

    assert np.array_equal(fhat["peaks_rowIx"], correct_peaks_rowIx)
    assert np.array_equal(fhat["peaks_colIx"], correct_peaks_colIx)

    return


def test_compute_centroids():
    data = np.array([[1, i] for i in range(4)])

    assert np.array_equal(compute_weighted_centroids(data),
                          np.array([1., ((1 + 2. + 3.) / 4.)]))

    return


def test_compute_weighted_centroids():
    data = np.array([[1, i] for i in range(4)])
    w = np.arange(4)
    w = w.reshape(w.shape[0], 1)

    assert np.array_equal(compute_weighted_centroids(data, w),
                          np.array([1., ((1 + 4 + 9) / 6.)]))

    return


def test_shrink_apert_no_weights():
    data = draw_gaussian(mean=np.ones(2), cov=np.eye(2), data_size=10000)
    shrink_peak = shrinking_apert(data)

    assert np.abs(shrink_peak[0] - 1.) < 5e-1  # performance may vary
    assert np.abs(shrink_peak[1] - 1.) < 5e-1

    return


def test_get_BCG_without_cuts():
    bands = ["r_band", "i_band"]

    # test_df has increasing magnitude, smaller is brighter
    test_df = pd.DataFrame(np.array([[i, j]
                                    for i in np.arange(1, 5)
                                    for j in np.arange(5, 10)]),
                           columns=bands)
    test_df["SubhaloLenType1"] = np.ones(test_df.shape[0])
    test_df["SubhaloLenType4"] = np.ones(test_df.shape[0])

    ix = get_BCG_ix(test_df, DM_cut=0, star_cut=0, bands=bands)

    # correct BCG is the first entry
    assert ix == 0

    return


# def test_get_py_peaks_and_density_weights():
#     """ Regression test """
#     x =
#     res = do_KDE_and_get_peaks(x)
#     fhat = convert_fhat_to_dict(res)
#
#     get_peaks(fhat)
#     # plot_KDE_peaks(fhat, allPeaks=True, showData=True)
#
#     return


# def test_weights_of_do_KDE_and_get_peaks():
#     py_x = np.array(x)
#     orig = np.array([-2, 2])
#     dist_x = np.array([np.sqrt(np.dot(x_row - orig, x_row - orig))
#                        for x_row in py_x])
#     w = np.ones(len(np.array(x)))
#     mask = dist_x < 0.3
#     w[mask] = 50.  # weight points near the (-2, 2) peak more
#
#     res = do_KDE_and_get_peaks(x, w=w)
#     fhat = convert_fhat_to_dict(res)
#     get_py_peaks_and_density_weights(fhat)
#     plot_KDE_peaks(fhat, allPeaks=True, showData=True)
#     return
