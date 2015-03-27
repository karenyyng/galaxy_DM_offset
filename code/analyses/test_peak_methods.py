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
                  cov=np.eye(2),
                  size=300):
    assert mean.shape[0] == cov.shape[0], "wrong mean and cov dimension"

    return np.random.multivariate_normal(mean, cov, size)


def one_big_gaussian_one_small_gaussian():
    gaussian1 = draw_gaussian(mean=np.zeros(2),
                              cov=np.eye(2) * 0.25, size=40)
    gaussian2 = draw_gaussian(mean=np.ones(2) * 2,
                              cov=np.eye(2), size=400)
    return np.vstack((gaussian1, gaussian2))


def dumbbell_data(cov_noise=np.array([[0.8, 0.5],
                                      [0.5, 0.8]]),
                  noise_wts=1 / 11., samp_no=400):

    gaussian1 = draw_gaussian(mean=np.ones(2) * 2,
                              cov=np.eye(2), size=250)
    gaussian2 = draw_gaussian(mean=-np.ones(2) * 2,
                              cov=np.eye(2), size=150)
    gaussian3 = draw_gaussian(mean=np.zeros(2),
                              cov=cov_noise, size=50)
    stuff = np.vstack((gaussian1, gaussian2))
    return np.vstack((stuff, gaussian3))


def mag_to_lum(I):
    """ I band magnitude is converted to luminosity
    the 23 in the exponent is put there for numerical stability
    """
    return np.exp(-I - 23.)


if __name__ == "__main__":
    g1 = draw_gaussian()
    plt.plot()
    pass
