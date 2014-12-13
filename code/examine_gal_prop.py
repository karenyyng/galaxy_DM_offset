"""identify galaxy properties in gal clusters from Illustris-1 simulation
Author: Karen Ng <karenyng@ucdavis.edu>
License: BSD
"""
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import rpy2.robjects as robjects


def plot_color_mag_diag(df, bluer_band, redder_band, band_limit,
                        plot=False, save=False, subhalo_len_lim=1e3,
                        savePath="../plots/", clst=None, verbose=False, *args,
                        **kwargs):
    """
    :parameters:
    df = pandas df of each cluster
    bluer_band = string, df colname
    red_band = string, df colname
    band_limit = float,
        how many band magnitude fainter than BCG do we want to examine
    particleLim = int,
        want to ensure subhalos has at least that many particles

    :returns: None

    :stability: works
    """
    bcg_i = df[redder_band].min()
    mask_i = df[redder_band] < bcg_i + band_limit

    if verbose:
        print "subhalos need at least {0} DM".format(subhalo_len_lim) + \
            " particles to be plotted"
    # examine number of DM particles
    mask_ii = df["SubhaloLenType1"] > subhalo_len_lim

    mask_i = np.logical_and(mask_i, mask_ii)

    g_i = df[bluer_band][mask_i] - df[redder_band][mask_i]

    plt.plot(df[redder_band][mask_i], g_i, "b.", alpha=0.3)
    plt.title("Color-magnitude diagram for".format(clst) +
              " {0} subhalos".format(np.sum(mask_i)))
    plt.ylabel(bluer_band + " - " + redder_band)
    plt.xlabel(redder_band)

    if clst is not None:
        plt.title("Cluster {0}: Color-magnitude diagram for".format(clst) +
              " {0} subhalos".format(np.sum(mask_i)))

    if save is True:
        assert clst is not None, "arg for clst missing"
        plt.savefig(savePath + "/cm_diagram{0}.png".format(clst),
                    bbox_inches="tight")
    plt.close()
    return


def plot_cf_contour(dens, x, y, lvls=[68, 95], show=False):
    """this sort through the density, add them up til they are
    below the required confidence level, then plot

    :param dens: = np.array, the density estimate, should integrate to 1
    :param x: = np.array, x coord of the density estimate
    :param y: = np.array, y coord of the density estimate
    :param lvls: = list of floats, denotes percentile
    """
    d = dens.ravel()
    lvls = np.array(lvls) / 100.
    lvl_vals = np.zeros(len(lvls))
    sums = 0

    d = np.sort(d)  # in ascending order
    d_sum = np.sum(d)

    for j in xrange(d.size):
        sums += d[j]
        for i in range(len(lvls)):
            if sums / d_sum <= 1. - lvls[i]:
                lvl_vals[i] = d[j]

    colors = [(0 / 255., 70 / 255., i / len(lvls)) for i in range(len(lvls))]

    # the plt.contour function is weird, if you don't transpose
    # the density, the plotted density will be rotated by 180 clockwise
    plt.contour(x, y, dens.transpose(), lvl_vals, linewidths=(2, 2),
                colors=colors)

    if show:
        plt.show()

    return


def plot_KDE_peaks(fhat, allpeaks=False, save=False,
                   fileName="./plots/py_KDE_peak_testcase_contours.png"):

    plt.axes().set_aspect('equal')
    plot_cf_contour(fhat["estimate"],
                    fhat["eval_points"][0], fhat["eval_points"][1],
                    lvls=range(0, 100, 10))

    plt.plot(fhat["data_x"][0], fhat["data_x"][1], 'k.', alpha=.4)
    plt.plot(fhat["domPeaks"].transpose()[0], fhat["domPeaks"].transpose()[1],
             'rx', mew=3, label='inferred dens peak')

    if allpeaks:
        for p in fhat["peaks_py_ix"]:
            plt.plot(fhat["eval_points"][0][p[0]], fhat["eval_points"][1][p[1]],
                     'bo', label='peaks', fillstyle='none', mew=1)

    plt.title("KDE of testcase by calling R from within python")

    plt.legend()

    plt.show()
    if save:
        plt.savefig(fileName, bbox_inches='tight')

    plt.close()
    return

