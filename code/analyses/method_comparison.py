""" constructs the Gaussian data sets to test the performance of different
centroiding methods

Author: Karen Ng
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../")
from get_gal_centroids import shrinking_apert, compute_weighted_mean


def draw_gaussian(mean=np.ones(2),
                  cov=np.array([[0.8, -0.3], [-0.3, 0.8]]),
                  size=300):
    assert mean.shape[0] == cov.shape[0], "wrong mean and cov dimension"

    return np.random.multivariate_normal(mean, cov, size)


def one_big_gaussian_one_small_gaussian():
    return


def twoD_bridged_data_set(cov_noise=np.array([[0.8, -0.72],
                                              [-0.72, 0.8]]),
                          noise_wts=1 / 11., samp_no=300):
    cov_diag = np.eye(2)


    return


if __name__ == "__main__":
    g1 = draw_gaussian()
    plt.plot()
    pass
