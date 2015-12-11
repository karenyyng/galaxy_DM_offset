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


def test_get_BCG_offsets_without_cuts():
    bands = ["r_band", "i_band"]

    # test_df has increasing magnitude, smaller is brighter
    test_df = pd.DataFrame(np.array([[i, j]
                                    for i in np.arange(1, 5)
                                    for j in np.arange(5, 10)]),
                           columns=bands)
    test_df["SubhaloLenType1"] = np.ones(test_df.shape[0])
    test_df["SubhaloLenType4"] = np.ones(test_df.shape[0])
    test_df["SubhaloPos0"] = np.arange(3, 3 + test_df.shape[0])
    test_df["SubhaloPos1"] = np.arange(6, 6 + test_df.shape[0])

    # correct BCG is the first entry
    assert get_BCG_offset(test_df, cut_kwargs={"DM_cut": 0, "star_cut": 0},
                          bands=bands) == np.sqrt(3. ** 2 + 6. ** 2)
    return


def test_rot_projection_of_project_coords():
    # to avoid numerical error due to 0 becomes very small number
    # we use allclose to check answers instead
    inputs = [(0, 0, 1), (1, 0, 0), (0, 1, 0)]
    ans = np.array([(0, 1, 0), (0, 0, -1), (-1, 0, 0)])

    for i, input_ele in enumerate(inputs):
        # not projecting anything so los_axis = 4
        # test ans from feeding each row of input
        assert np.allclose(project_coords(input_ele, 90, 90, 4, radian=False),
                           np.array(ans[i]))

    # test vectorization
    test_outputs = project_coords(inputs, 90, 90, 4, radian=False)

    assert np.allclose(ans, test_outputs)
    return


def test_project_to_lower_dim_of_project_coords():
    """
    :tasks: should test values that are not multiples of 90
    """
    los_axes = [2, 1, 0]
    inputs = [(0, 0, 1), (1, 0, 0), (0, 1, 0)]
    ans = [(0, 1, 0), (0, 0, -1), (0, 0, 0)]
    for i, input_ele in enumerate(inputs):
        assert np.allclose(project_coords(input_ele, 90, 90, los_axes[i],
                                          radian=False),
                           np.array(ans[i]))
    return


def test_galaxies_closest_to_peak():
    col = [0, 1]
    df = pd.DataFrame([[i, i] for i in np.arange(10)], columns=col)
    peak_coords = [(3.25, 3.5), (1., 1.5)]

    dist, ixes = galaxies_closest_to_peak(df, col, peak_coords,
                                          k_nearest_neighbor=1)
    # KDTree.query returns an integer if k_nearest_neighbor = 1
    assert ixes[0] == 3
    assert ixes[1] == 1

    return


def test_angles_given_HEALpix_nsides():
    """regression test"""
    xi = np.array([ 0.84106867,  0.84106867,  1.57079633,  1.57079633,
                    2.30052398, 2.30052398]),
    phi = np.array([ 0.78539816,  2.35619449,  0.        ,  1.57079633,
                    0.78539816, 2.35619449])

    test_xi, test_phi = angles_given_HEALpix_nsides(1)
    assert np.allclose(xi, test_xi)
    assert np.allclose(phi, test_phi)


def test_angles_from_HEALpix_nsides_should_not_show_same_projection():
    from itertools import combinations

    xi_arr, phi_arr = angles_given_HEALpix_nsides(1)

    # The number of comparisons needed is nC2,
    combinations(range(len(xi_arr)), 2)

    assert np.sum([

    ])


def test_same_projection():
    """
    :returns: TODO

    """
    coords_w_same_projection = np.array((0, 0, 180, 0)) * np.pi / 180.
    coords_not_of_same_projection = np.array((0, 0, 90, 0)) * np.pi / 180.

    assert same_projection(*coords_w_same_projection) == True
    assert same_projection(*coords_not_of_same_projection) == False
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
