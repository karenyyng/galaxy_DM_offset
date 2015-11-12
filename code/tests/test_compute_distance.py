import sys
sys.path.append("../")
from compute_distance import *
import numpy as np


def test_compute_distance_between_DM_and_gal_peaks():
    fhat_stars = {"peaks_xcoords": np.array([0.1, 1., 10.]) * 75. / 106.5,
                  "peaks_ycoords": np.array([0.5, 5.,  5.]) * 75. / 106.5,
                  "peaks_dens": np.array([1., .1, .2])
                  }
    fhat = {"peaks_xcoords": np.array([1.1, 1., 10., 3., 20, 16, 2]),
            "peaks_ycoords": np.array([0.5, 5.,  5., 4., 10, 17, 2]),
            "peaks_dens": np.array([1., .7, .2, .7, .6, .1, .8])
            }

    (dist, ixes), gal_peaks_no, DM_peak_no = \
        compute_distance_between_DM_and_gal_peaks(fhat_stars, fhat)

    assert dist[0] == 1.0, \
        "dist between coordinates are not correct."

    assert ixes[0] == 0, \
        "nearest neighbor ixes are not correct."

    assert gal_peaks_no == 1, \
        "gal_peaks_no is not correct."

    assert DM_peak_no == 6, \
        "DM_peak_no is not correct."

    return

