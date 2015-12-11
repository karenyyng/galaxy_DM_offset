import sys
sys.path.append("../")
from compute_distance import *
import numpy as np


def test_compute_distance_between_DM_and_gal_peaks():
    conversion_factor = 0.704
    fhat_stars = {"peaks_xcoords": np.array([0.1, 1., 10.]) * conversion_factor,
                  "peaks_ycoords": np.array([0.5, 5.,  5.]) * conversion_factor,
                  "peaks_dens": np.array([1., .1, .2])
                  }
    fhat = {"peaks_xcoords": np.array([1.1, 1., 10., 3., 20, 16, 2]),
            "peaks_ycoords": np.array([0.5, 5.,  5., 4., 10, 17, 2]),
            "peaks_dens": np.array([1., .7, .2, .7, .6, .1, .8])
            }

    (dist, ixes), gal_peaks_no, DM_peak_no, good_threshold = \
        compute_distance_between_DM_and_gal_peaks(fhat_stars, fhat)

    assert np.allclose(dist[0], 1.0), \
        "dist between coordinates are not correct."

    assert ixes[0] == 0, \
        "nearest neighbor ixes are not correct."

    assert gal_peaks_no == 1, \
        "gal_peaks_no is not correct."

    assert DM_peak_no == 6, \
        "DM_peak_no is not correct."

    return


def test_compute_euclidean_dist():
    """test if we are computing correct distances
    """
    test_pt1 = (1, 1)
    test_pt2 = (1, 1, 1)
    test_pts = np.array([[2, 1, 1], [1, 3, 1], [1, 1, 4]])

    assert compute_euclidean_dist(test_pt1) == np.sqrt(2), \
        "Failed computing norm of {}".format(test_p1)

    assert compute_euclidean_dist(test_pt2) == np.sqrt(3), \
        "Failed computing norm of {}".format(test_p2)

    assert np.array_equal(compute_euclidean_dist(test_pts, test_pt2),
                          np.array([1, 2, 3])), \
                          "Failed computing distance of {0} and {1}".format(test_pts, test_pts)
