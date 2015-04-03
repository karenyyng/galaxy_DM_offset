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
import get_gal_centroids as get_gal
from plot_gal_prop import plot_cf_contour


def draw_gaussian(mean=np.ones(2),
                  cov=np.eye(2),
                  data_size=300):
    assert mean.shape[0] == cov.shape[0], "wrong mean and cov dimension"

    return np.random.multivariate_normal(mean, cov, data_size)


def call_gaussian_and_prepare_data(data_size=500, bootNo=100):
    gauss_data = [draw_gaussian(mean=np.ones(2),
                                cov=np.eye(2),
                                data_size=data_size)
                  for i in range(bootNo)]

    # get shrinking aperture peak
    shrink_peak_dens = get_gal.get_shrinking_apert_conf_reg(gauss_data)

    # get KDE peaks
    KDE_peak_dens = get_gal.get_KDE_conf_reg(gauss_data)

    # get centroid method confidence region estimate
    cent_peak_dens = get_gal.get_centroid_conf_reg(gauss_data)

    return gauss_data, shrink_peak_dens, KDE_peak_dens, cent_peak_dens


def one_big_gaussian_one_small_gaussian(data_size=500,
                                        mixture_fraction=0.3):
    # the smaller gaussian is made to be a bit more compact
    gaussian1 = draw_gaussian(mean=np.zeros(2),
                              cov=np.eye(2) * mixture_fraction ** .5,
                              data_size=int(data_size * mixture_fraction))
    gaussian2 = draw_gaussian(mean=np.ones(2) * 2,
                              cov=np.eye(2),
                              data_size=int(data_size *
                                            (1. - mixture_fraction)))
    return np.vstack((gaussian1, gaussian2))


def dumbbell_data(cov_noise=np.array([[0.8, 0.5],
                                      [0.5, 0.8]]),
                  data_size=500,
                  mixture_fraction=(.55, .35, .10)):
    gaussian1 = draw_gaussian(mean=np.ones(2) * 2.,
                              cov=np.eye(2),
                              data_size=int(
                                  data_size * mixture_fraction[0]))
    gaussian2 = draw_gaussian(mean=np.ones(2) * -2.,
                              cov=np.eye(2),
                              data_size=int(
                                  data_size * mixture_fraction[1]))
    gaussian3 = draw_gaussian(mean=np.zeros(2),
                              cov=cov_noise,
                              data_size=int(
                                  data_size * mixture_fraction[2]))
    stuff = np.vstack((gaussian1, gaussian2))
    return np.vstack((stuff, gaussian3))


def call_dumbbell_example_and_prepare_data(data_size=500, bootNo=100,
                                           mixture_fraction=(.55, .35, .1),
                                           set_seed=True):
    if set_seed:
        np.random.seed(1)

    dumb_data = [dumbbell_data(mixture_fraction=(.55, .35, .1),
                               data_size=data_size)
                 for i in range(bootNo)]

    # get shrinking aperture peak
    shrink_peak_dens2 = get_gal.get_shrinking_apert_conf_reg(dumb_data)

    # get KDE peaks
    KDE_peak_dens2, KDE_peak_dens2b = \
        get_gal.get_KDE_conf_reg(dumb_data, second_peak=True)

    # get centroid method confidence region estimate
    cent_peak_dens2 = get_gal.get_centroid_conf_reg(dumb_data)

    return dumb_data, shrink_peak_dens2, KDE_peak_dens2, KDE_peak_dens2b,\
        cent_peak_dens2


def plot_dumbbell_500_comparison(
        dumb_data,
        shrink_peak_dens2,
        KDE_peak_dens2,
        KDE_peak_dens2b,
        cent_peak_dens2,
        figsidesize=7,
        plot_path="../../paper/figures/drafts/",
        plot_fig_name="confidence_regions_dumbbell_500.eps"):

    # plot styles
    lvls = [68, 95]
    g_colors = [(0 / 255., i / (len(lvls) + 1.), 0 / 255.)
                for i in range(1, len(lvls) + 1)]

    b_colors = [(0 / 255., 0 / 255., i / (len(lvls) + 1.))
                for i in range(1, len(lvls) + 1)]

    r_colors = [(i / (len(lvls) + 1.), 0 / 255., 0 / 255.)
                for i in range(1, len(lvls) + 1)]

    plt.figure(figsize=(figsidesize * 3, figsidesize))

    # right most subplot
    plt.subplot(133)
    markersize = 10
    # plot KDE dominant peak contour
    plot_cf_contour(KDE_peak_dens2["estimate"],
                    KDE_peak_dens2["eval_points"][0],
                    KDE_peak_dens2["eval_points"][1],
                    colors=b_colors)

    # plot KDE subdominant peak contour
    plt.plot(KDE_peak_dens2["peaks_xcoords"][0],
             KDE_peak_dens2["peaks_ycoords"][0],
             'bx', mew=2, markersize=markersize, fillstyle='none',
             label="KDE peak best est")

    # plot shrinking aperture contour
    plot_cf_contour(shrink_peak_dens2["estimate"],
                    shrink_peak_dens2["eval_points"][0],
                    shrink_peak_dens2["eval_points"][1],
                    colors=g_colors)

    plt.plot(shrink_peak_dens2["peaks_xcoords"][0],
             shrink_peak_dens2["peaks_ycoords"][0],
             'gx', mew=2, markersize=markersize,
             label="Shrink peak best est")

    plt.plot(2, 2, "kx", mew=2, label="Mean of dominant Gaussian",
             markersize=markersize)

    plt.legend(loc='lower right', frameon=False)
    plt.title("Zoomed-in view near the dominant peak",
              fontsize=15)

    plt.xlim(0.25, 3.0)
    plt.ylim(0.25, 3.0)

    # middle subplot
    plt.subplot(132)
    markersize = 6
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)

    plot_cf_contour(KDE_peak_dens2["estimate"],
                    KDE_peak_dens2["eval_points"][0],
                    KDE_peak_dens2["eval_points"][1],
                    colors=b_colors)
    plt.figtext(0.55, 0.53, 'KDE dominant\npeak confidence \nregion',
                color='b')

    plot_cf_contour(KDE_peak_dens2b["estimate"],
                    KDE_peak_dens2b["eval_points"][0],
                    KDE_peak_dens2b["eval_points"][1],
                    colors=b_colors)
    plt.figtext(0.49, 0.35, 'KDE subdominant peak\nconfidence region',
                color='b')

    plot_cf_contour(shrink_peak_dens2["estimate"],
                    shrink_peak_dens2["eval_points"][0],
                    shrink_peak_dens2["eval_points"][1],
                    colors=g_colors)

    plt.figtext(0.45, 0.65, 'Shrink apert peak\nconfidence region',
                color='g')

    plot_cf_contour(cent_peak_dens2["estimate"],
                    cent_peak_dens2["eval_points"][0],
                    cent_peak_dens2["eval_points"][1],
                    colors=r_colors)
    plt.figtext(0.43, 0.55, 'Centroid \nconfidence region', color='r')

    markersize = 8
    plt.plot(2, 2, "kx", mew=2,
             label="Mean of dominant Gaussian",
             markersize=markersize)
    plt.plot(-2, -2, "x", color="grey", mew=3,
             label="Mean of subdominant Gaussian",
             markersize=markersize)
    plt.plot(0, 0, "x", color="grey", mew=3,
             label="Mean of subdominant Gaussian",
             markersize=markersize)
    plt.legend(loc='lower right', frameon=False)

    plt.title('Confidence regions and best estimates of peak finding methods',
              fontsize=13)

    # first plot for the data
    plt.subplot(131)
    plt.plot(dumb_data[0][:, 0], dumb_data[0][:, 1], '.', color='grey')
    plt.plot(2, 2, "ko", mew=2,
             label="Mean of dominant Gaussian", fillstyle='none',
             markersize=markersize)
    plt.plot(-2, -2, "x", color="k", mew=3,
             label="Mean of subdominant Gaussian",
             markersize=markersize)
    plt.plot(0, 0, "x", color="k", mew=3,
             label="Mean of subdominant Gaussian",
             markersize=markersize)
    plt.legend(loc='best', frameon=False)
    plt.title("Dumbbell data with 3 mixtures of Gaussians", size=15)
    plt.savefig(plot_path + plot_fig_name, bbox_inches='tight')
    return


def plot_dumbbell_50_comparison(
        dumb_data,
        shrink_peak_dens2,
        KDE_peak_dens2,
        KDE_peak_dens2b,
        cent_peak_dens2,
        figsidesize=7,
        plot_path="../../paper/figures/drafts/",
        plot_fig_name="confidence_regions_dumbbell_50.eps"):

    # plot styles
    lvls = [68, 95]
    g_colors = [(0 / 255., i / (len(lvls) + 1.), 0 / 255.)
                for i in range(1, len(lvls) + 1)]

    b_colors = [(0 / 255., 0 / 255., i / (len(lvls) + 1.))
                for i in range(1, len(lvls) + 1)]

    r_colors = [(i / (len(lvls) + 1.), 0 / 255., 0 / 255.)
                for i in range(1, len(lvls) + 1)]

    plt.figure(figsize=(figsidesize * 3, figsidesize))

    # right most subplot
    plt.subplot(133)
    markersize = 10
    # plot KDE dominant peak contour
    plot_cf_contour(KDE_peak_dens2["estimate"],
                    KDE_peak_dens2["eval_points"][0],
                    KDE_peak_dens2["eval_points"][1],
                    colors=b_colors)

    # plot KDE subdominant peak contour
    plt.plot(KDE_peak_dens2["peaks_xcoords"][0],
             KDE_peak_dens2["peaks_ycoords"][0],
             'bx', mew=2, markersize=markersize, fillstyle='none',
             label="KDE peak best est")

    # plot shrinking aperture contour
    plot_cf_contour(shrink_peak_dens2["estimate"],
                    shrink_peak_dens2["eval_points"][0],
                    shrink_peak_dens2["eval_points"][1],
                    colors=g_colors)

    plt.plot(shrink_peak_dens2["peaks_xcoords"][0],
             shrink_peak_dens2["peaks_ycoords"][0],
             'gx', mew=2, markersize=markersize,
             label="Shrink peak best est")

    plt.plot(2, 2, "kx", mew=2, label="Mean of dominant Gaussian",
             markersize=markersize)

    plt.legend(loc='lower right', frameon=False)
    plt.title("Zoomed-in view near the dominant peak",
              fontsize=15)

    plt.xlim(0.25, 3.0)
    plt.ylim(0.25, 3.0)

    # middle subplot
    plt.subplot(132)
    markersize = 6
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)

    plot_cf_contour(KDE_peak_dens2["estimate"],
                    KDE_peak_dens2["eval_points"][0],
                    KDE_peak_dens2["eval_points"][1],
                    colors=b_colors)
    plt.figtext(0.55, 0.53, 'KDE dominant\npeak confidence \nregion',
                color='b')

    plot_cf_contour(KDE_peak_dens2b["estimate"],
                    KDE_peak_dens2b["eval_points"][0],
                    KDE_peak_dens2b["eval_points"][1],
                    colors=b_colors)

    plt.figtext(0.49, 0.35, 'KDE subdominant peak\nconfidence region',
                color='b')

    plot_cf_contour(shrink_peak_dens2["estimate"],
                    shrink_peak_dens2["eval_points"][0],
                    shrink_peak_dens2["eval_points"][1],
                    colors=g_colors)

    plt.figtext(0.45, 0.65, 'Shrink apert peak\nconfidence region',
                color='g')

    plot_cf_contour(cent_peak_dens2["estimate"],
                    cent_peak_dens2["eval_points"][0],
                    cent_peak_dens2["eval_points"][1],
                    colors=r_colors)
    plt.figtext(0.43, 0.55, 'Centroid \nconfidence region', color='r')

    markersize = 8
    plt.plot(2, 2, "kx", mew=2,
             label="Mean of dominant Gaussian",
             markersize=markersize)
    plt.plot(-2, -2, "x", color="grey", mew=3,
             label="Mean of subdominant Gaussian",
             markersize=markersize)
    plt.plot(0, 0, "x", color="grey", mew=3,
             label="Mean of subdominant Gaussian",
             markersize=markersize)
    plt.legend(loc='lower right', frameon=False)

    plt.title('Confidence regions and best estimates of peak finding methods',
              fontsize=13)

    # first plot for the data
    plt.subplot(131)
    plt.plot(dumb_data[0][:, 0], dumb_data[0][:, 1], '.', color='grey')
    plt.plot(2, 2, "ko", mew=2,
             label="Mean of dominant Gaussian", fillstyle='none',
             markersize=markersize)
    plt.plot(-2, -2, "x", color="k", mew=3,
             label="Mean of subdominant Gaussian",
             markersize=markersize)
    plt.plot(0, 0, "x", color="k", mew=3,
             label="Mean of subdominant Gaussian",
             markersize=markersize)
    plt.legend(loc='best', frameon=False)
    plt.title("Dumbbell data with 3 mixtures of Gaussians", size=15)
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)

    plt.savefig(plot_path + plot_fig_name, bbox_inches='tight')

    return


if __name__ == "__main__":
    # g1 = draw_gaussian()
    # plt.plot()
    pass
