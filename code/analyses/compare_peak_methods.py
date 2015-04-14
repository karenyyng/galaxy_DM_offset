""" constructs the Gaussian data sets to test the performance of different
centroiding methods

Author: Karen Ng
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import OrderedDict
import matplotlib.pyplot as ax
import cPickle
import numpy as np
import sys
sys.path.append("../")
import get_gal_centroids as get_gal
from plot_gal_prop import plot_cf_contour


# plot styles
lvls = [68, 95]
g_colors = [(0 / 255., i / (len(lvls) + 1.), 0 / 255.)
            for i in xrange(1, len(lvls) + 1)]

b_colors = [(0 / 255., 0 / 255., i / (len(lvls) + 1.))
            for i in xrange(1, len(lvls) + 1)]

r_colors = [(i / (len(lvls) + 1.), 0 / 255., 0 / 255.)
            for i in xrange(1, len(lvls) + 1)]

# ------------------data sampling -----------------------------

def draw_gaussian(mean=np.ones(2),
                  cov=np.eye(2),
                  data_size=300):
    assert mean.shape[0] == cov.shape[0], "wrong mean and cov dimension"

    return np.random.multivariate_normal(mean, cov, data_size)


def call_gaussian_and_prepare_data(data_size=500, bootNo=100, save=False,
                                   path="../../data/"):
    d = OrderedDict()
    gausskey = "gauss" + str(data_size)
    d[gausskey] = [draw_gaussian(mean=np.ones(2),
                                 cov=np.eye(2),
                                 data_size=data_size)
                   for i in range(bootNo)]

    # get shrinking aperture peak
    d["shrink"] = get_gal.get_shrinking_apert_conf_reg(d[gausskey])

    # get KDE peaks
    d["KDE"] = get_gal.get_KDE_conf_reg(d[gausskey])

    # get centroid method confidence region estimate
    d["cent"] = get_gal.get_centroid_conf_reg(d[gausskey])

    if save:
        save_data(d, path)

    return d


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


def call_one_big_one_small_gaussian(data_size=500, bootNo=100, save=False,
                                    path="../../data/"):
    d = OrderedDict()
    bimodalkey = "bimodal" + str(data_size)
    d[bimodalkey] = \
        [one_big_gaussian_one_small_gaussian(data_size=data_size)
         for i in range(bootNo)]

    d["shrink"] = get_gal.get_shrinking_apert_conf_reg(d[bimodalkey])

    d["KDE"] = get_gal.get_KDE_conf_reg(d[bimodalkey])

    d["cent"] = get_gal.get_centroid_conf_reg(d[bimodalkey])

    if save:
        save_data(d, path)

    return d


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
                                           set_seed=True, save=False,
                                           path="../../data"):
    if set_seed:
        np.random.seed(1)

    d = OrderedDict()

    dumbkey = "dumb" + str(data_size)
    d[dumbkey] = \
        [dumbbell_data(mixture_fraction=mixture_fraction,
                       data_size=data_size)
         for i in range(bootNo)]

    # get shrinking aperture peak
    d["shrink"] = get_gal.get_shrinking_apert_conf_reg(d[dumbkey])

    # get KDE peaks
    d["KDE1"], d["KDE2"] = \
        get_gal.get_KDE_conf_reg(d[dumbkey], second_peak=True)

    # get centroid method confidence region estimate
    d["cent"] = get_gal.get_centroid_conf_reg(d[dumbkey])

    if save:
        save_data(d, path)

    return d

# -----------------plotting functions ----------------------


def save_data(d, path="../../data/"):
    dataDetail = d.keys()[0]
    for k, v in d.iteritems():
        if k == dataDetail:
            filename = path + k + ".pkl"
        else:
            filename = path + k + "_" + dataDetail + ".pkl"
        print(filename)
        cPickle.dump(v, open(filename, "w"))
    return


def plot_one_big_one_small_gaussian_500(
        ax, bimodal_data, shrink_peak_dens1, KDE_peak_dens1, cent_peak_dens1,
        figsize=7, fig_path="../../paper/figures/drafts/",
        fig_name="confidence_regions_bimodal_500.pdf"):

    ax.figure(figsize=(figsize * 3, figsize))
    ax.subplot(131)
    ax.plot(bimodal_data[0][:, 0], bimodal_data[0][:, 1], 'k.', alpha=0.3)
    ax.plot(2, 2, 'kx', mew=2, ms=10, label='Mean of dominant Gaussian')
    ax.plot(0, 0, 'x', color='grey',
            mew=2, ms=10, label='Mean of subdominant Gaussian')
    ax.legend(loc='best', frameon=False)

    return


def peak_est_contours_one_big_one_small_gaussian(
        ax, KDE_peak_dens1, shrink_peak_dens1, cent_peak_dens1, xlim, ylim):
    plot_cf_contour(KDE_peak_dens1["estimate"],
                    KDE_peak_dens1["eval_points"][0],
                    KDE_peak_dens1["eval_points"][1],
                    colors=b_colors)
    ax.annotate('KDE peak\nconfidence region', (0.12, 0.52),
                textcoords='axes fraction',
                color='b')

    plot_cf_contour(shrink_peak_dens1["estimate"],
                    shrink_peak_dens1["eval_points"][0],
                    shrink_peak_dens1["eval_points"][1],
                    colors=g_colors)
    ax.annotate('Shrink. apert. peak\nconfidence region', (0.3, 0.7),
                textcoords='axes fraction',
                color='g')

    plot_cf_contour(cent_peak_dens1["estimate"],
                    cent_peak_dens1["eval_points"][0],
                    cent_peak_dens1["eval_points"][1],
                    colors=r_colors)
    ax.annotate('Centroid\nconfidence region', (0.3, 0.42),
                textcoords='axes fraction',
                color='r')

    ax.plot(2, 2, "kx", mew=2, label="True center", markersize=5)
    ax.legend(loc='best', frameon=False)
    # ax.title('Confidence region from one Gaussian at (1, 1)',
    #           fontsize=15)
    ax.xlim(xlim)
    ax.ylim(ylim)
    return


def zoomed_in_view(ax, KDE_peak_dens, shrink_peak_dens1, cent_peak_dens1,
                   xlim, ylim, markersize):
    markersize = 10
    # plot KDE dominant peak contour
    plot_cf_contour(KDE_peak_dens1["estimate"],
                    KDE_peak_dens1["eval_points"][0],
                    KDE_peak_dens1["eval_points"][1],
                    colors=b_colors)

    ax.plot(KDE_peak_dens1["peaks_xcoords"][0],
            KDE_peak_dens1["peaks_ycoords"][0],
            'bx', mew=2, markersize=markersize,
            label="KDE peak best est")

    # plot shrinking aperture contour
    plot_cf_contour(shrink_peak_dens1["estimate"],
                    shrink_peak_dens1["eval_points"][0],
                    shrink_peak_dens1["eval_points"][1],
                    colors=g_colors)

    ax.plot(shrink_peak_dens1["peaks_xcoords"][0],
            shrink_peak_dens1["peaks_ycoords"][0],
            'gx', mew=2, markersize=markersize,
            label="Shrink peak best est")

    # plot centroid contour
    plot_cf_contour(cent_peak_dens1["estimate"],
                    cent_peak_dens1["eval_points"][0],
                    cent_peak_dens1["eval_points"][1],
                    colors=r_colors)

    ax.plot(cent_peak_dens1["peaks_xcoords"][0],
            cent_peak_dens1["peaks_ycoords"][0],
            'rx', mew=2, markersize=markersize,
            label="Centroid peak best est")

    ax.plot(2, 2, "kx", mew=2, label="Mean of dominant Gaussian",
            markersize=markersize)

    ax.ylim(0, 3)

    ax.legend(loc='lower right', frameon=False)
    ax.title("Zoomed-in view near the dominant peak",
             fontsize=15)

    print("saving figure to" + fig_path + fig_name)
    ax.savefig(fig_path + fig_name, bbox_inches='tight')

    return


def plot_one_big_one_small_gaussian(
        bimodal_data, shrink_peak_dens1, KDE_peak_dens1, cent_peak_dens1,
        figsize=7, fig_path="../../paper/figures/drafts/",
        fig_name="confidence_regions_bimodal.pdf", save=False):

    ax.figure(figsize=(figsize * 3, figsize))
    ax.subplot(131)
    ax.plot(bimodal_data[0][:, 0], bimodal_data[0][:, 1], 'k.', alpha=0.3)
    ax.plot(2, 2, 'kx', mew=2, ms=10, label='Mean of dominant Gaussian')
    ax.plot(0, 0, 'x', color='grey',
            mew=2, ms=10, label='Mean of subdominant Gaussian')
    ax.legend(loc='best', frameon=False)

    xlim = ax.xlim(-2, 5)
    ylim = ax.ylim(-2, 5)

    ax.subplot(132)

    plot_cf_contour(KDE_peak_dens1["estimate"],
                    KDE_peak_dens1["eval_points"][0],
                    KDE_peak_dens1["eval_points"][1],
                    colors=b_colors)
    ax.annotate('KDE peak\nconfidence region', (0.12, 0.52),
                textcoords='axes fraction',
                color='b')

    plot_cf_contour(shrink_peak_dens1["estimate"],
                    shrink_peak_dens1["eval_points"][0],
                    shrink_peak_dens1["eval_points"][1],
                    colors=g_colors)
    ax.annotate('Shrink. apert. peak\nconfidence region', (0.3, 0.7),
                textcoords='axes fraction',
                color='g')

    plot_cf_contour(cent_peak_dens1["estimate"],
                    cent_peak_dens1["eval_points"][0],
                    cent_peak_dens1["eval_points"][1],
                    colors=r_colors)
    ax.annotate('Centroid\nconfidence region', (0.3, 0.42),
                textcoords='axes fraction',
                color='r')

    ax.plot(2, 2, "kx", mew=2, label="True center", markersize=5)
    ax.legend(loc='best', frameon=False)
    # ax.title('Confidence region from one Gaussian at (1, 1)',
    #           fontsize=15)
    ax.xlim(xlim)
    ax.ylim(ylim)

    ax.subplot(133)
    markersize = 10
    # plot KDE dominant peak contour
    plot_cf_contour(KDE_peak_dens1["estimate"],
                    KDE_peak_dens1["eval_points"][0],
                    KDE_peak_dens1["eval_points"][1],
                    colors=b_colors)

    ax.plot(KDE_peak_dens1["peaks_xcoords"][0],
            KDE_peak_dens1["peaks_ycoords"][0],
            'bx', mew=2, markersize=markersize,
            label="KDE peak best est")

    # plot shrinking aperture contour
    plot_cf_contour(shrink_peak_dens1["estimate"],
                    shrink_peak_dens1["eval_points"][0],
                    shrink_peak_dens1["eval_points"][1],
                    colors=g_colors)

    ax.plot(shrink_peak_dens1["peaks_xcoords"][0],
            shrink_peak_dens1["peaks_ycoords"][0],
            'gx', mew=2, markersize=markersize,
            label="Shrink peak best est")

    # plot centroid contour
    plot_cf_contour(cent_peak_dens1["estimate"],
                    cent_peak_dens1["eval_points"][0],
                    cent_peak_dens1["eval_points"][1],
                    colors=r_colors)

    ax.plot(cent_peak_dens1["peaks_xcoords"][0],
            cent_peak_dens1["peaks_ycoords"][0],
            'rx', mew=2, markersize=markersize,
            label="Centroid peak best est")

    ax.plot(2, 2, "kx", mew=2, label="Mean of dominant Gaussian",
            markersize=markersize)

    ax.ylim(0, 3)

    ax.legend(loc='lower right', frameon=False)
    ax.title("Zoomed-in view near the dominant peak",
             fontsize=15)

    if save:
        print("saving figure to" + fig_path + fig_name)
        ax.savefig(fig_path + fig_name, bbox_inches='tight')

    return


def plot_dumbbell_500_comparison(
        dumbbell_data2,
        shrink_peak_dens2,
        KDE_peak_dens2,
        KDE_peak_dens2b,
        cent_peak_dens2,
        figsidesize=7,
        plot_path="../../paper/figures/drafts/",
        plot_fig_name="confidence_regions_dumbbell_500.pdf"):
    """ for plot in appendix, tweaked labels to look best on fig"""

    ax.figure(figsize=(figsidesize * 3, figsidesize))

    # right most subplot
    ax.subplot(133)
    markersize = 10
    # plot KDE dominant peak contour
    plot_cf_contour(KDE_peak_dens2["estimate"],
                    KDE_peak_dens2["eval_points"][0],
                    KDE_peak_dens2["eval_points"][1],
                    colors=b_colors)

    # plot KDE subdominant peak contour
    ax.plot(KDE_peak_dens2["peaks_xcoords"][0],
            KDE_peak_dens2["peaks_ycoords"][0],
            'bx', mew=2, markersize=markersize, fillstyle='none',
            label="KDE peak best est")

    # plot shrinking aperture contour
    plot_cf_contour(shrink_peak_dens2["estimate"],
                    shrink_peak_dens2["eval_points"][0],
                    shrink_peak_dens2["eval_points"][1],
                    colors=g_colors)

    ax.plot(shrink_peak_dens2["peaks_xcoords"][0],
            shrink_peak_dens2["peaks_ycoords"][0],
            'gx', mew=2, markersize=markersize,
            label="Shrink peak best est")

    ax.plot(2, 2, "kx", mew=2, label="Mean of dominant Gaussian",
            markersize=markersize)

    ax.legend(loc='lower right', frameon=False)
    ax.title("Zoomed-in view near the dominant peak",
             fontsize=15)

    ax.xlim(0.25, 3.0)
    ax.ylim(0.25, 3.0)

    # middle subplot
    ax.subplot(132)
    markersize = 6
    ax.xlim(-6, 6)
    ax.ylim(-6, 6)

    plot_cf_contour(KDE_peak_dens2["estimate"],
                    KDE_peak_dens2["eval_points"][0],
                    KDE_peak_dens2["eval_points"][1],
                    colors=b_colors)
    ax.figtext(0.55, 0.53, 'KDE dominant\npeak confidence \nregion',
               color='b')

    plot_cf_contour(KDE_peak_dens2b["estimate"],
                    KDE_peak_dens2b["eval_points"][0],
                    KDE_peak_dens2b["eval_points"][1],
                    colors=b_colors)
    ax.figtext(0.49, 0.35, 'KDE subdominant peak\nconfidence region',
               color='b')

    plot_cf_contour(shrink_peak_dens2["estimate"],
                    shrink_peak_dens2["eval_points"][0],
                    shrink_peak_dens2["eval_points"][1],
                    colors=g_colors)

    ax.figtext(0.45, 0.65, 'Shrink apert peak\nconfidence region',
               color='g')

    plot_cf_contour(cent_peak_dens2["estimate"],
                    cent_peak_dens2["eval_points"][0],
                    cent_peak_dens2["eval_points"][1],
                    colors=r_colors)
    ax.figtext(0.43, 0.55, 'Centroid \nconfidence region', color='r')

    markersize = 8
    ax.plot(2, 2, "kx", mew=2,
            label="Mean of dominant Gaussian",
            markersize=markersize)
    ax.plot(-2, -2, "x", color="grey", mew=3,
            label="Mean of subdominant Gaussian",
            markersize=markersize)
    ax.plot(0, 0, "x", color="grey", mew=3,
            label="Mean of subdominant Gaussian",
            markersize=markersize)
    ax.legend(loc='lower right', frameon=False)

    ax.title('Confidence regions and best estimates of peak finding methods',
             fontsize=13)

    # first plot for the data
    ax.subplot(131)
    ax.plot(d["dumb"][0][:, 0], dumb_data[0][:, 1], '.', color='k',
            alpha=0.3)
    ax.plot(2, 2, "ko", mew=2,
            label="Mean of dominant Gaussian", fillstyle='none',
            markersize=markersize)
    ax.plot(-2, -2, "x", color="k", mew=3,
            label="Mean of subdominant Gaussian",
            markersize=markersize)
    ax.plot(0, 0, "x", color="k", mew=3,
            label="Mean of subdominant Gaussian",
            markersize=markersize)
    ax.legend(loc='best', frameon=False)
    ax.title("Dumbbell data with 3 mixtures of Gaussians", size=15)
    ax.savefig(plot_path + plot_fig_name, bbox_inches='tight')
    return


def plot_dumbbell_comparison(
        ax,
        dumb_data,
        shrink_peak_dens2,
        KDE_peak_dens2,
        KDE_peak_dens2b,
        cent_peak_dens2,
        figsidesize=7,
        save=False,
        plot_path="../../paper/figures/drafts/",
        plot_fig_name="confidence_regions_dumbbell.pdf"):

    ax.figure(figsize=(figsidesize * 3, figsidesize))

    # right most subplot
    ax.subplot(133)
    markersize = 10
    # plot KDE dominant peak contour
    plot_cf_contour(KDE_peak_dens2["estimate"],
                    KDE_peak_dens2["eval_points"][0],
                    KDE_peak_dens2["eval_points"][1],
                    colors=b_colors)

    # plot KDE subdominant peak contour
    ax.plot(KDE_peak_dens2["peaks_xcoords"][0],
            KDE_peak_dens2["peaks_ycoords"][0],
            'bx', mew=2, markersize=markersize, fillstyle='none',
            label="KDE peak best est")

    # plot shrinking aperture contour
    plot_cf_contour(shrink_peak_dens2["estimate"],
                    shrink_peak_dens2["eval_points"][0],
                    shrink_peak_dens2["eval_points"][1],
                    colors=g_colors)

    ax.plot(shrink_peak_dens2["peaks_xcoords"][0],
            shrink_peak_dens2["peaks_ycoords"][0],
            'gx', mew=2, markersize=markersize,
            label="Shrink peak best est")

    ax.plot(2, 2, "kx", mew=2, label="Mean of dominant Gaussian",
            markersize=markersize)

    ax.legend(loc='lower right', frameon=False)
    ax.title("Zoomed-in view near the dominant peak",
             fontsize=15)

    ax.xlim(0.25, 3.0)
    ax.ylim(0.25, 3.0)

    # middle subplot
    ax.subplot(132)
    markersize = 6
    ax.xlim(-6, 6)
    ax.ylim(-6, 6)

    plot_cf_contour(KDE_peak_dens2["estimate"],
                    KDE_peak_dens2["eval_points"][0],
                    KDE_peak_dens2["eval_points"][1],
                    colors=b_colors)
    ax.figtext(0.55, 0.53, 'KDE dominant\npeak confidence \nregion',
               color='b')

    plot_cf_contour(KDE_peak_dens2b["estimate"],
                    KDE_peak_dens2b["eval_points"][0],
                    KDE_peak_dens2b["eval_points"][1],
                    colors=b_colors)

    ax.figtext(0.49, 0.35, 'KDE subdominant peak\nconfidence region',
               color='b')

    plot_cf_contour(shrink_peak_dens2["estimate"],
                    shrink_peak_dens2["eval_points"][0],
                    shrink_peak_dens2["eval_points"][1],
                    colors=g_colors)

    ax.figtext(0.45, 0.65, 'Shrink apert peak\nconfidence region',
               color='g')

    plot_cf_contour(cent_peak_dens2["estimate"],
                    cent_peak_dens2["eval_points"][0],
                    cent_peak_dens2["eval_points"][1],
                    colors=r_colors)
    ax.figtext(0.43, 0.55, 'Centroid \nconfidence region', color='r')

    markersize = 8
    ax.plot(2, 2, "kx", mew=2,
            label="Mean of dominant Gaussian",
            markersize=markersize)
    ax.plot(-2, -2, "x", color="grey", mew=3,
            label="Mean of subdominant Gaussian",
            markersize=markersize)
    ax.plot(0, 0, "x", color="grey", mew=3,
            label="Mean of subdominant Gaussian",
            markersize=markersize)
    ax.legend(loc='lower right', frameon=False)

    ax.title('Confidence regions and best estimates of peak finding methods',
             fontsize=13)

    # first plot for the data
    ax.subplot(131)
    ax.plot(dumb_data[0][:, 0], dumb_data[0][:, 1], '.', color='grey')
    ax.plot(2, 2, "ko", mew=2,
            label="Mean of dominant Gaussian", fillstyle='none',
            markersize=markersize)
    ax.plot(-2, -2, "x", color="k", mew=3,
            label="Mean of subdominant Gaussian",
            markersize=markersize)
    ax.plot(0, 0, "x", color="k", mew=3,
            label="Mean of subdominant Gaussian",
            markersize=markersize)
    ax.legend(loc='best', frameon=False)
    ax.title("Dumbbell data with 3 mixtures of Gaussians", size=15)
    ax.xlim(-6, 6)
    ax.ylim(-6, 6)

    if save:
        print("saving figure to" + plot_path + plot_fig_name)
        ax.savefig(plot_path + plot_fig_name, bbox_inches='tight')

    return


def plot_gauss_500_comparison(gauss_data, shrink_peak_dens, KDE_peak_dens,
                              cent_peak_dens,
                              fig_path="../../paper/figures/drafts/",
                              fig_name="gauss500.pdf",
                              figsize=7, save=False):

    ax.figure(figsize=(figsize * 3, figsize))
    ax.subplot(131)
    ax.plot(d["gauss"][0][:, 0], gauss_data[0][:, 1], 'k.', alpha=0.3)
    ax.plot(1, 1, 'kx', mew=2, ms=10, label='Mean of Gaussian')
    ax.legend(loc='best', frameon=False)

    xlim = ax.xlim(-2, 4)
    ylim = ax.ylim(-2, 4)

    ax.subplot(132)

    plot_cf_contour(KDE_peak_dens["estimate"],
                    KDE_peak_dens["eval_points"][0],
                    KDE_peak_dens["eval_points"][1],
                    colors=b_colors)
    ax.annotate('KDE peak\nconfidence region', (0.3, 0.62),
                textcoords='axes fraction',
                color='b')

    plot_cf_contour(shrink_peak_dens["estimate"],
                    shrink_peak_dens["eval_points"][0],
                    shrink_peak_dens["eval_points"][1],
                    colors=g_colors)
    ax.annotate('Shrink. apert. peak\nconfidence region', (0.3, 0.25),
                textcoords='axes fraction',
                color='g')

    plot_cf_contour(cent_peak_dens["estimate"],
                    cent_peak_dens["eval_points"][0],
                    cent_peak_dens["eval_points"][1],
                    colors=r_colors)
    ax.annotate('Centroid\nconfidence region', (0.6, 0.5),
                textcoords='axes fraction',
                color='r')
    ax.plot(1, 1, "kx", mew=2, label="True center", markersize=5)
    ax.xlim(0, 2.0)
    ax.legend(loc='best', frameon=False)
    # ax.title('Confidence region from one Gaussian at (1, 1)',
    #           fontsize=15)
    ax.xlim(xlim)
    ax.ylim(ylim)

    ax.subplot(133)
    markersize = 10
    # plot KDE dominant peak contour
    plot_cf_contour(KDE_peak_dens["estimate"],
                    KDE_peak_dens["eval_points"][0],
                    KDE_peak_dens["eval_points"][1],
                    colors=b_colors)

    ax.plot(KDE_peak_dens["peaks_xcoords"][0],
            KDE_peak_dens["peaks_ycoords"][0],
            'bx', mew=2, markersize=markersize,
            label="KDE peak best est")

    # plot shrinking aperture contour
    plot_cf_contour(shrink_peak_dens["estimate"],
                    shrink_peak_dens["eval_points"][0],
                    shrink_peak_dens["eval_points"][1],
                    colors=g_colors)

    ax.plot(shrink_peak_dens["peaks_xcoords"][0],
            shrink_peak_dens["peaks_ycoords"][0],
            'gx', mew=2, markersize=markersize,
            label="Shrink peak best est")

    # plot centroid contour
    plot_cf_contour(cent_peak_dens["estimate"],
                    cent_peak_dens["eval_points"][0],
                    cent_peak_dens["eval_points"][1],
                    colors=r_colors)

    ax.plot(cent_peak_dens["peaks_xcoords"][0],
            cent_peak_dens["peaks_ycoords"][0],
            'rx', mew=2, markersize=markersize,
            label="Centroid peak best est")

    ax.plot(1, 1, "kx", mew=2, label="Mean of dominant Gaussian",
            markersize=markersize)

    ax.legend(loc='lower right', frameon=False)
    ax.title("Zoomed-in view near the dominant peak",
             fontsize=15)

    if save:
        print("saving figure to" + fig_path + fig_name)
        ax.savefig(fig_path + fig_name, bbox_inches='tight')

    return


def plot_gauss_comparison(gauss_data, shrink_peak_dens, KDE_peak_dens,
                          cent_peak_dens,
                          fig_path="../../paper/figures/drafts/",
                          fig_name="gauss.pdf",
                          figsize=7, save=False):

    ax.figure(figsize=(figsize * 3, figsize))
    ax.subplot(131)
    ax.plot(d["gauss"][0][:, 0], gauss_data[0][:, 1], 'k.', alpha=0.3)
    ax.plot(1, 1, 'kx', mew=2, ms=10, label='Mean of Gaussian')
    ax.legend(loc='best', frameon=False)

    xlim = ax.xlim(-2, 4)
    ylim = ax.ylim(-2, 4)

    ax.subplot(132)

    plot_cf_contour(KDE_peak_dens["estimate"],
                    KDE_peak_dens["eval_points"][0],
                    KDE_peak_dens["eval_points"][1],
                    colors=b_colors)
    ax.annotate('KDE peak\nconfidence region', (0.3, 0.62),
                textcoords='axes fraction',
                color='b')

    plot_cf_contour(shrink_peak_dens["estimate"],
                    shrink_peak_dens["eval_points"][0],
                    shrink_peak_dens["eval_points"][1],
                    colors=g_colors)
    ax.annotate('Shrink. apert. peak\nconfidence region', (0.3, 0.25),
                textcoords='axes fraction',
                color='g')

    plot_cf_contour(cent_peak_dens["estimate"],
                    cent_peak_dens["eval_points"][0],
                    cent_peak_dens["eval_points"][1],
                    colors=r_colors)
    ax.annotate('Centroid\nconfidence region', (0.6, 0.5),
                textcoords='axes fraction',
                color='r')
    ax.plot(1, 1, "kx", mew=2, label="True center", markersize=5)
    ax.xlim(0, 2.0)
    ax.legend(loc='best', frameon=False)
    # ax.title('Confidence region from one Gaussian at (1, 1)',
    #           fontsize=15)
    ax.xlim(xlim)
    ax.ylim(ylim)

    ax.subplot(133)
    markersize = 10
    # plot KDE dominant peak contour
    plot_cf_contour(KDE_peak_dens["estimate"],
                    KDE_peak_dens["eval_points"][0],
                    KDE_peak_dens["eval_points"][1],
                    colors=b_colors)

    ax.plot(KDE_peak_dens["peaks_xcoords"][0],
            KDE_peak_dens["peaks_ycoords"][0],
            'bx', mew=2, markersize=markersize,
            label="KDE peak best est")

    # plot shrinking aperture contour
    plot_cf_contour(shrink_peak_dens["estimate"],
                    shrink_peak_dens["eval_points"][0],
                    shrink_peak_dens["eval_points"][1],
                    colors=g_colors)

    ax.plot(shrink_peak_dens["peaks_xcoords"][0],
            shrink_peak_dens["peaks_ycoords"][0],
            'gx', mew=2, markersize=markersize,
            label="Shrink peak best est")

    # plot centroid contour
    plot_cf_contour(cent_peak_dens["estimate"],
                    cent_peak_dens["eval_points"][0],
                    cent_peak_dens["eval_points"][1],
                    colors=r_colors)

    ax.plot(cent_peak_dens["peaks_xcoords"][0],
            cent_peak_dens["peaks_ycoords"][0],
            'rx', mew=2, markersize=markersize,
            label="Centroid peak best est")

    ax.plot(1, 1, "kx", mew=2, label="Mean of dominant Gaussian",
            markersize=markersize)

    ax.legend(loc='lower right', frameon=False)
    ax.title("Zoomed-in view near the dominant peak",
             fontsize=15)

    if save:
        print("saving figure to" + fig_path + fig_name)
        ax.savefig(fig_path + fig_name, bbox_inches='tight')

    return


def infer_CR():
    return
