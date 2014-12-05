"""identify galaxy properties in gal clusters from Illustris-1 simulation
Author: Karen Ng <karenyng@ucdavis.edu>
License: BSD
"""
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


def plot_KDE_peaks(fhat, save=False,
               fileName="./plots/py_KDE_peak_testcase_contours.png"):

    plt.axes().set_aspect('equal')
    plt.contour(fhat["eval_points"][0], fhat["eval_points"][1],
                fhat["estimate"], label='dens contour')

    plt.plot(fhat["data_x"][0], fhat["data_x"][1], 'k.', alpha=.4,
             label='data')
    plt.plot(fhat["domPeaks"].transpose()[0], fhat["domPeaks"].transpose()[1],
             'rx', mew=3, label='inferred dens peak')
    plt.title("KDE of testcase by calling R from within python")

    plt.legend()


    if save:
        plt.savefig(fileName, bbox_inches='tight')

    plt.close()
    return


def contour_plot():
    return
