""" constructs the Gaussian data sets to test the performance of different
centroiding methods

Author: Karen Ng <karenyng@ucdavis.edu>
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import OrderedDict
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import sys
sys.path.append("../")
import get_gal_centroids as get_gal
from plot_gal_prop import plot_cf_contour
rc("font", family="serif")


# plot styles
# lvls = [68]  #, 95]
# g_colors = [(0 / 255., i / (len(lvls) + 1.), 0 / 255.)
#             for i in xrange(1, len(lvls) + 1)]
# b_colors = [(0 / 255., 0 / 255., i / (len(lvls) + 1.))
#             for i in xrange(1, len(lvls) + 1)]
# r_colors = [(i / (len(lvls) + 1.), 0 / 255., 0 / 255.)
#             for i in xrange(1, len(lvls) + 1)]

# Color palette are from http://colorbrewer2.org/
g_colors = [(127./255, 191./255, 123./255), (27./255, 120/255., 55./255)]

b_colors = [(103./255, 169./255, 207./255), (33./255, 102/255., 172./255)]

# This represents purple to be color-blind friendly
r_colors = [(136./255, 86./255, 167./255), (129./255, 15./255, 124./255)]

# ------------------data sampling -----------------------------

def draw_gaussian(mean=np.ones(2),
                  cov=np.eye(2),
                  data_size=300):
    assert mean.shape[0] == cov.shape[0], "wrong mean and cov dimension"

    return np.random.multivariate_normal(mean, cov, data_size)


def call_gaussian_and_prepare_data(data_size=500, bootNo=100, save=False,
                                   path="../../data/"):
    d = OrderedDict()
    gausskey = "gauss"  # + str(data_size)
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
        save_data_to_h5(d, data_size, path)

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
    bimodalkey = "bimodal"  # + str(data_size)
    d[bimodalkey] = \
        [one_big_gaussian_one_small_gaussian(data_size=data_size)
         for i in range(bootNo)]

    d["shrink"] = get_gal.get_shrinking_apert_conf_reg(d[bimodalkey])

    d["KDE"] = get_gal.get_KDE_conf_reg(d[bimodalkey])

    d["cent"] = get_gal.get_centroid_conf_reg(d[bimodalkey])

    if save:
        save_data_to_h5(d, data_size, path)

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

    dumbkey = "dumb"  # + str(data_size)
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
        save_data_to_h5(d, data_size, path)

    return d


def save_data_to_pkl(d, path="../../data/"):
    dataDetail = d.keys()[0]
    for k, v in d.iteritems():
        if k == dataDetail:
            filename = path + k + ".pkl"
        else:
            filename = path + k + "_" + dataDetail + ".pkl"
        print(filename)
        cPickle.dump(v, open(filename, "w"))
    return


def save_data_to_h5(d, data_size, path="../../data", h5="compare_methods.h5"):
    """ to be debugged"""
    import h5py
    dataDetail = d.keys()[0]
    f = h5py.File(path + h5, "a", compression="gzip", complevel=9)
    try:
        gp = f.create_group(str(data_size))
    except ValueError:
        gp = f[str(data_size)]

    try:
        gp1 = gp.create_group(dataDetail)
    except ValueError:
        gp1 = gp[dataDetail]

    for k, v in d.iteritems():
        if k == dataDetail:
            gp1["data"] = v
        else:
            gp1.create_group(k)
            fhat = v
            for fhat_key, fhat_val in fhat.iteritems():
                gp1[k + "/" + fhat_key] = fhat_val

            # get_gal.convert_dict_peaks_to_df(gp1[k], wt="", save=False)

    f.close()
    return

# -----------------plotting functions ----------------------


def read_in_data_for_left_3_cols(folder_path="../../data/fig2_data/",
                                 data_size=50):
    """handle the data to be plotted"""
    import os
    pkl_lists = os.listdir(folder_path)

    methods = ["KDE", "shrink", "cent"]
    dumb_methods = ["KDE1", "KDE2", "shrink", "cent"]

    # pick out file names for suitable sets of data
    gauss_pkls = \
        [p for p in pkl_lists if "_gauss" + str(data_size) + ".pkl" in p]
    bimodal_pkls = \
        [p for p in pkl_lists if "_bimodal" + str(data_size) + ".pkl" in p]
    dumb_pkls = \
        [p for p in pkl_lists if "_dumb" + str(data_size) + ".pkl" in p]
    print(dumb_pkls)

    # load the suitable files based on the paths, preserving order of the
    # methods in the methods list
    gauss_data = OrderedDict(
        {m: cPickle.load(open(folder_path + p)) for p in gauss_pkls
        for m in methods if m in p})

    bi_data = OrderedDict(
        {m: cPickle.load(open(folder_path + p)) for p in bimodal_pkls
         for m in methods if m in p})

    dumb_data = OrderedDict(
        {m: cPickle.load(open(folder_path + p))
         for m in dumb_methods for p in dumb_pkls if m in p})

    # only want the first set of the sampled data
    gauss_data["data"] = cPickle.load(open(folder_path +
                                           "gauss" + str(data_size) +
                                           ".pkl"))[0]
    bi_data["data"] = cPickle.load(open(folder_path +
                                   "bimodal" + str(data_size) +
                                   ".pkl"))[0]
    dumb_data["data"] = cPickle.load(open(folder_path +
                                     "dumb" + str(data_size) +
                                     ".pkl"))[0]

    return gauss_data, bi_data, dumb_data


def plot_grid_spec(gauss_data, bimodal_data, dumb_data, f, figsize=(13, 13),
                   save=False, output_path="../../paper/figures/drafts/",
                   name="fig2.eps"):
    """
    :param gauss_data: hdf5 hierarchial data structure
    :param bimodal_data: hdf5 hierarchial data structure
    :param dumb_data: hdf5 hierarchial data structure
    :param f: hdf5 hierarchical file stream
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MaxNLocator

    plt.figure(figsize=figsize)
    rowNo = 3
    colNo1 = 1
    colNo2 = 2
    hspace = 0.1

    # plot the left two columns
    gs1 = gridspec.GridSpec(rowNo, colNo1)
    gs1.update(left=0.05, right=0.44, wspace=0.01, hspace=hspace)
    axArr1 = [[plt.subplot(gs1[i, j], aspect='equal')
               for j in range(colNo1)] for i in range(rowNo)]

    # plot the right two columns
    gs2 = gridspec.GridSpec(rowNo, colNo2)
    gs2.update(left=0.51, right=0.98, wspace=0.4, hspace=hspace)
    axArr2 = [[plt.subplot(gs2[i, j])
               for j in range(colNo2)] for i in range(rowNo)]

    # plt.setp([axArr1[j][0].get_yticklabels()
    #           for j in range(rowNo)], visible=False)

    # for j in range(rowNo):
    #     axArr1[j][0].xaxis.set_major_locator(
    #         MaxNLocator(nbins=5, prune="lower"))

    # first row of plots
    xlim, ylim = plot_gauss_data(gauss_data["data"], ax=axArr1[0][0])
    # plot_gauss_contour(gauss_data["KDE"],
    #                    gauss_data["shrink"],
    #                    gauss_data["cent"],
    #                    ax=axArr1[0][1], xlim=xlim,
    #                    ylim=ylim)
    plot_gauss_zoomed_contours(gauss_data["KDE"],
                               gauss_data["shrink"],
                               gauss_data["cent"],
                               ax=axArr2[0][0])

    # second row of plots
    xlim, ylim = plot_one_big_one_small_gaussian(
        bimodal_data["data"], ax=axArr1[1][0],
        xlim=(-2, 4), ylim=(-2, 4)
    )
    # plot_one_big_one_small_gaussian_contour(bimodal_data["KDE"],
    #                                         bimodal_data["shrink"],
    #                                         bimodal_data["cent"],
    #                                         xlim=xlim, ylim=ylim,
    #                                         ax=axArr1[1][1])
    plot_one_big_one_small_gaussian_zoomed_contour(
                                                   bimodal_data["KDE"],
                                                   bimodal_data["shrink"],
                                                   bimodal_data["cent"],
                                                   ax=axArr2[1][0])

    # third row of plots
    xlim, ylim = plot_dumbbell_data(dumb_data["data"], ax=axArr1[2][0],
                                    xlim=(-4, 4), ylim=(-4, 4))
    # plot_dumbbell_contour(dumb_data["KDE1"],
    #                       dumb_data["KDE2"],
    #                       dumb_data["shrink"],
    #                       dumb_data["cent"],
    #                       ax=axArr1[2][1],
    #                       xlim=xlim, ylim=ylim)
    plot_dumbbell_zoomed_contour(dumb_data["KDE1"],
                                 dumb_data["KDE2"],
                                 dumb_data["shrink"],
                                 dumb_data["cent"],
                                 ax=axArr2[2][0])

    # first row rightmost column
    plot_error_as_a_func_of_data_pts(f, "gauss",
                                     ax=axArr2[0][1])
    plot_error_as_a_func_of_data_pts(f, "bimodal",
                                     ax=axArr2[1][1])
    plot_error_as_a_func_of_data_pts(f, "dumb",
                                     ax=axArr2[2][1],
                                     methods=["KDE1", "shrink", "cent"])

    plt.savefig(output_path + name, bbox_inches='tight')

    return


def plot_error_as_a_func_of_data_pts(f, data_set,
                                     ax=None, methods=["KDE",
                                                       "shrink",
                                                       "cent"],
                                     show_xlabel=False
                                     ):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    data_size = [20, 50, 100, 500]

    # if len(colors) != len(methods):
    #     raise ValueError("Number of color supplied for plotting must\n" +
    #                      "match number of peak finding methods")

    errs = compute_errors_from_data(f, data_set, methods=methods,
                                    data_size=data_size)

    for i, method in enumerate(methods):
        cf68 = [errs[str(size)][method].values()[0]
                for size in data_size]
        ax.plot(data_size, cf68, 'x-',
                mew=2, ms=10,
                label=method)

    ax.set_xlim(0, 550)
    ax.set_ylim(0, ax.get_ylim()[-1] * 1.1)
    ax.set_ylabel("68% confidence region", size=15)
    if show_xlabel:
        ax.set_xlabel("Number of data points", size="small")

    ax.tick_params(labeltop='off', labelright='off')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    return


def compute_errors_from_data(f, data_set, methods,
                             data_size=[20, 50, 100, 500]):


    errs = {}
    for size in data_size:
        errs[str(size)] = {}
        gp = f[str(size) + "/" + data_set]
        for method in methods:
            errs[str(size)][method] = {}
            contours = plot_cf_contour(gp[method]["estimate"][:],
                                       gp[method]["eval_points"][:][0],
                                       gp[method]["eval_points"][:][1])
            plt.clf()
            plt.close()
            peaks = np.array([gp[method]["peaks_xcoords"][:][0],
                              gp[method]["peaks_ycoords"][:][0]])
            for k, ctr in contours.iteritems():
                errs[str(size)][method][k] = calculate_error_from_contour(ctr,
                                                                         peaks)

    return errs


def calculate_error_from_contour(contour, peak):
    err = np.array([np.sqrt(np.dot(diff, diff)) for diff in contour - peak])
    return np.median(err)


def plot_gauss_data(gauss_data, ax=None, xlim=None,
                    ylim=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    ax.plot(gauss_data[:][0][:, 0], gauss_data[:][0][:, 1], 'k.', alpha=0.3)
    ax.plot(1, 1, 'kx', mew=2, ms=10, label='Mean of Gaussian')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(labeltop='off', labelright='off')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    return ax.get_xlim(), ax.get_ylim()


def plot_gauss_contour(KDE_peak_dens, shrink_peak_dens,
                       cent_peak_dens, ax=None, xlim=None,
                       ylim=None,
                       b_colors=b_colors, g_colors=g_colors, r_colors=r_colors
                       ):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

    contours = {}
    plot_cf_contour(KDE_peak_dens["estimate"][:],
                    KDE_peak_dens["eval_points"][:][0],
                    KDE_peak_dens["eval_points"][:][1],
                    colors=b_colors, ax=ax)
    # ax.annotate('KDE peak\nconfidence region', (0.3, 0.62),
    #             textcoords='axes fraction',
    #             color='b')

    plot_cf_contour(shrink_peak_dens["estimate"][:],
                    shrink_peak_dens["eval_points"][:][0],
                    shrink_peak_dens["eval_points"][:][1],
                    colors=g_colors, ax=ax)
    # ax.annotate('Shrink. apert. peak\nconfidence region', (0.3, 0.25),
    #             textcoords='axes fraction',
    #             color='g')

    plot_cf_contour(cent_peak_dens["estimate"][:],
                    cent_peak_dens["eval_points"][:][0],
                    cent_peak_dens["eval_points"][:][1],
                    colors=r_colors, ax=ax)
    # ax.annotate('Centroid\nconfidence region', (0.6, 0.5),
    #             textcoords='axes fraction',
    #             color='r')

    ax.plot(1, 1, "kx", mew=2, label="True center", markersize=5)
    ax.legend(loc='best', frameon=False)
    # ax.title('Confidence region from one Gaussian at (1, 1)',
    #           fontsize=15)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.tick_params(labeltop='off', labelright='off')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    return


def plot_gauss_zoomed_contours(KDE_peak_dens1, shrink_peak_dens1,
                               cent_peak_dens1, xlim=None, ylim=None,
                               markersize=10,
                               b_colors=b_colors, g_colors=g_colors,
                               r_colors=r_colors,
                               ax=None):
    """
    :param KDE_peak_dens1: dictionary like / hdf5 file stream with appropriate
        keys that correspond to outputs from the r KDE function
    :param shrink_peak_dens1: same as above but for shrinking aperture data
    :param cent_peak_dens1: see above

    :return contours: dictionary of dictionaries of contour positions
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # plot KDE dominant peak contour
    plot_cf_contour(KDE_peak_dens1["estimate"][:],
                    KDE_peak_dens1["eval_points"][:][0],
                    KDE_peak_dens1["eval_points"][:][1],
                    colors=b_colors, ax=ax)

    ax.plot(KDE_peak_dens1["peaks_xcoords"][:][0],
            KDE_peak_dens1["peaks_ycoords"][:][0],
            'bx', mew=2, markersize=markersize,
            label="KDE peak best est")

    # plot shrinking aperture contour
    plot_cf_contour(shrink_peak_dens1["estimate"][:],
                    shrink_peak_dens1["eval_points"][:][0],
                    shrink_peak_dens1["eval_points"][:][1],
                    colors=g_colors, ax=ax)

    ax.plot(shrink_peak_dens1["peaks_xcoords"][:][0],
            shrink_peak_dens1["peaks_ycoords"][:][0],
            'gx', mew=2, markersize=markersize,
            label="Shrink peak best est")

    # plot centroid contour
    plot_cf_contour(cent_peak_dens1["estimate"][:],
                    cent_peak_dens1["eval_points"][:][0],
                    cent_peak_dens1["eval_points"][:][1],
                    colors=r_colors, ax=ax)

    ax.plot(cent_peak_dens1["peaks_xcoords"][:][0],
            cent_peak_dens1["peaks_ycoords"][:][0],
            'rx', mew=2, markersize=markersize,
            label="Centroid peak best est")

    ax.plot(1, 1, "kx", mew=2, label="Mean of dominant Gaussian",
            markersize=markersize)

    if ylim is not None:
        ax.set_ylim(ylim)

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.tick_params(labeltop='off', labelright='off')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # ax.legend(loc='lower right', frameon=False, fontsize='small')
    # ax.title("Zoomed-in view near the dominant peak",
    #          fontsize=15)

    # if save:
    #     print("saving figure to" + fig_path + fig_name)
    #     ax.savefig(fig_path + fig_name, bbox_inches='tight')

    return


def plot_one_big_one_small_gaussian(
        bimodal_data, figsize=7, fig_path="../../paper/figures/drafts/",
        fig_name="confidence_regions_bimodal.pdf", save=False, ax=None,
        xlim=None, ylim=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    ax.plot(bimodal_data[:][0][:, 0], bimodal_data[:][0][:, 1], 'k.', alpha=0.3)
    ax.plot(2, 2, 'kx', mew=2, ms=10, label='Mean of dominant Gaussian')
    ax.plot(0, 0, 'x', color='grey',
            mew=2, ms=10, label='Mean of subdominant Gaussian')
    # ax.legend(loc='best', frameon=False, fontsize='small')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.tick_params(labeltop='off', labelright='off')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    return ax.get_xlim(), ax.get_ylim()


def plot_one_big_one_small_gaussian_contour(
        KDE_peak_dens1, shrink_peak_dens1, cent_peak_dens1,
        figsize=7, xlim=None, ylim=None,
        fig_path="../../paper/figures/drafts/",
        fig_name="confidence_regions_bimodal.pdf", save=False, ax=None):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    plot_cf_contour(KDE_peak_dens1["estimate"][:],
                    KDE_peak_dens1["eval_points"][:][0],
                    KDE_peak_dens1["eval_points"][:][1],
                    colors=b_colors, ax=ax)
    # ax.annotate('KDE peak\nconfidence region', (0.12, 0.52),
    #             textcoords='axes fraction',
    #             color='b')

    plot_cf_contour(shrink_peak_dens1["estimate"][:],
                    shrink_peak_dens1["eval_points"][:][0],
                    shrink_peak_dens1["eval_points"][:][1],
                    colors=g_colors, ax=ax)
    # ax.annotate('Shrink. apert. peak\nconfidence region', (0.3, 0.7),
    #             textcoords='axes fraction',
    #             color='g')

    plot_cf_contour(cent_peak_dens1["estimate"][:],
                    cent_peak_dens1["eval_points"][:][0],
                    cent_peak_dens1["eval_points"][:][1],
                    colors=r_colors, ax=ax)
    # ax.annotate('Centroid\nconfidence region', (0.3, 0.42),
    #             textcoords='axes fraction',
    #             color='r')

    ax.plot(2, 2, "kx", mew=2, label="True center", markersize=5)
    # ax.legend(loc='best', frameon=False)
    # ax.title('Confidence region from one Gaussian at (1, 1)',
    #           fontsize=15)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.tick_params(labeltop='off', labelright='off')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    return


def plot_one_big_one_small_gaussian_zoomed_contour(
        KDE_peak_dens1, shrink_peak_dens1, cent_peak_dens1,
        figsize=7, fig_path="../../paper/figures/drafts/", markersize=10,
        fig_name="confidence_regions_bimodal.pdf", save=False, ax=None,
        xlim=None, ylim=None):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    contours = {}
    # plot KDE dominant peak contour
    contours["KDE"] = plot_cf_contour(KDE_peak_dens1["estimate"][:],
                                      KDE_peak_dens1["eval_points"][:][0],
                                      KDE_peak_dens1["eval_points"][:][1],
                                      colors=b_colors, ax=ax)

    ax.plot(KDE_peak_dens1["peaks_xcoords"][:][0],
            KDE_peak_dens1["peaks_ycoords"][:][0],
            'bx', mew=2, markersize=markersize,
            label="KDE peak best est")

    # plot shrinking aperture contour
    contours["shrink"] = plot_cf_contour(shrink_peak_dens1["estimate"][:],
                                         shrink_peak_dens1["eval_points"][:][0],
                                         shrink_peak_dens1["eval_points"][:][1],
                                         colors=g_colors, ax=ax)

    ax.plot(shrink_peak_dens1["peaks_xcoords"][:][0],
            shrink_peak_dens1["peaks_ycoords"][:][0],
            'gx', mew=2, markersize=markersize,
            label="Shrink peak best est")

    # plot centroid contour
    contours["cent"] = plot_cf_contour(cent_peak_dens1["estimate"][:],
                                       cent_peak_dens1["eval_points"][:][0],
                                       cent_peak_dens1["eval_points"][:][1],
                                       colors=r_colors, ax=ax)

    ax.plot(cent_peak_dens1["peaks_xcoords"][:][0],
            cent_peak_dens1["peaks_ycoords"][:][0],
            'rx', mew=2, markersize=markersize,
            label="Centroid peak best est")

    ax.plot(2, 2, "kx", mew=2, label="Mean of dominant Gaussian",
            markersize=markersize)

    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)

    ax.tick_params(labeltop='off', labelright='off')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # ax.legend(loc='lower right', frameon=False, fontsize='small')
    return contours


def plot_dumbbell_data(
        dumb_data, xlim=None, ylim=None, save=False,
        plot_path="../../paper/figures/drafts/", markersize=10,
        plot_fig_name="confidence_regions_dumbbell.pdf", ax=None):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    ax.plot(dumb_data[:][0][:, 0], dumb_data[:][0][:, 1], '.', color='grey')
    ax.plot(2, 2, "kx", mew=3,
            label="Mean of dominant Gaussian", fillstyle='none',
            markersize=markersize)
    ax.plot(-2, -2, "x", color="grey", mew=3,
            markersize=markersize)
    ax.plot(0, 0, "x", color="grey", mew=3,
            label="Mean of subdominant Gaussian",
            markersize=markersize)
    # ax.title("Dumbbell data with 3 mixtures of Gaussians", size=15)
    ax.legend(loc='lower right', frameon=True, bbox_to_anchor=(1., -0.4))

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.tick_params(labeltop='off', labelright='off')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    return ax.get_xlim(), ax.get_ylim()


def plot_dumbbell_contour(
        KDE_peak_dens2,
        KDE_peak_dens2b,
        shrink_peak_dens2,
        cent_peak_dens2,
        xlim=None, ylim=None,
        save=False, markersize=6,
        plot_path="../../paper/figures/drafts/",
        plot_fig_name="confidence_regions_dumbbell.pdf", ax=None,
        r_colors=r_colors, b_colors=b_colors, g_colors=g_colors
    ):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    plot_cf_contour(KDE_peak_dens2["estimate"][:],
                    KDE_peak_dens2["eval_points"][:][0],
                    KDE_peak_dens2["eval_points"][:][1],
                    colors=b_colors, ax=ax)
    # ax.annotate('KDE dominant\npeak confidence region', (0.55, 0.53),
    #             textcoords='axes fraction',
    #             color='b')

    plot_cf_contour(KDE_peak_dens2b["estimate"][:],
                    KDE_peak_dens2b["eval_points"][:][0],
                    KDE_peak_dens2b["eval_points"][:][1],
                    colors=b_colors, ax=ax)

    # ax.annotate('KDE subdominant\npeak confidence region', (0.49, 0.35),
    #             textcoords='axes fraction',
    #             color='b')

    plot_cf_contour(shrink_peak_dens2["estimate"][:],
                    shrink_peak_dens2["eval_points"][:][0],
                    shrink_peak_dens2["eval_points"][:][1],
                    colors=g_colors, ax=ax)

    # ax.annotate('Shrink apert peak\nconfidence region', (0.45, 0.65),
    #             textcoords='axes fraction',
    #             color='g')

    plot_cf_contour(cent_peak_dens2["estimate"][:],
                    cent_peak_dens2["eval_points"][:][0],
                    cent_peak_dens2["eval_points"][:][1],
                    colors=r_colors, ax=ax)
    # ax.annotate('Centroid \nconfidence region', (0.43, 0.55),
    #             textcoords='axes fraction',
    #             color='r')

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

    ax.tick_params(labeltop='off', labelright='off')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # ax.legend(loc='lower right', frameon=False, fontsize='small')

    # ax.title('Confidence regions and best estimates of peak finding methods',
    #          fontsize=13)

    return


def plot_dumbbell_zoomed_contour(
        KDE_peak_dens2,
        KDE_peak_dens2b,
        shrink_peak_dens2,
        cent_peak_dens2, markersize=10,
        xlim=(0.25, 4), ylim=(-0.6, 4),
        save=False,
        plot_path="../../paper/figures/drafts/",
        plot_fig_name="confidence_regions_dumbbell.pdf", ax=None,
        b_colors=b_colors, r_colors=r_colors, g_colors=g_colors
    ):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    markersize = 10
    contours = {}
    # plot KDE dominant peak contour
    contours["KDE"] = plot_cf_contour(KDE_peak_dens2["estimate"][:],
                                      KDE_peak_dens2["eval_points"][:][0],
                                      KDE_peak_dens2["eval_points"][:][1],
                                      colors=b_colors, ax=ax)

    # plot KDE subdominant peak contour
    ax.plot(KDE_peak_dens2["peaks_xcoords"][:][0],
            KDE_peak_dens2["peaks_ycoords"][:][0],
            'bx', mew=2, markersize=markersize, fillstyle='none',
            label="KDE peak best est")

    # plot shrinking aperture contour
    contours["shrink"] = plot_cf_contour(shrink_peak_dens2["estimate"][:],
                                         shrink_peak_dens2["eval_points"][:][0],
                                         shrink_peak_dens2["eval_points"][:][1],
                                         colors=g_colors, ax=ax)

    ax.plot(shrink_peak_dens2["peaks_xcoords"][:][0],
            shrink_peak_dens2["peaks_ycoords"][:][0],
            'gx', mew=2, markersize=markersize,
            label="Shrink peak best est")

    ax.plot(2, 2, "kx", mew=2, label="Mean of dominant Gaussian",
            markersize=markersize)

    ax.tick_params(labeltop='off', labelright='off')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
        # ax.legend(loc='lower right', frameon=False, fontsize='small')
    # ax.title("Zoomed-in view near the dominant peak",
    #          fontsize=15)

    return contours
# ------------previous drafts ----------------------------------

def plot_compare_one_big_one_small_gaussian(bimodal_data,
        KDE_peak_dens1, shrink_peak_dens1, cent_peak_dens1,
        figsize=7, fig_path="../../paper/figures/drafts/",
        fig_name="confidence_regions_bimodal.pdf", save=False, ax=None):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

    ax.plot(bimodal_data[0][:, 0], bimodal_data[0][:, 1], 'k.', alpha=0.3)
    ax.plot(2, 2, 'kx', mew=2, ms=10, label='Mean of dominant Gaussian')
    ax.plot(0, 0, 'x', color='grey',
            mew=2, ms=10, label='Mean of subdominant Gaussian')
    ax.legend(loc='best', frameon=False)

    xlim = ax.xlim(-2, 5)
    ylim = ax.ylim(-2, 5)

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


def peak_est_contours_one_big_one_small_gaussian(
        KDE_peak_dens1, shrink_peak_dens1, cent_peak_dens1, xlim, ylim,
        ax=None, save=False, figname="bimodal_CR_contour.pdf"):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

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
    ax.xlim(xlim)
    ax.ylim(ylim)

    if save:
        ax.savefig(figname, bbox_inches='tight')
    return


def plot_one_big_one_small_gaussian_500(
        bimodal_data, shrink_peak_dens1, KDE_peak_dens1, cent_peak_dens1,
        figsize=7, ax=None, fig_path="../../paper/figures/drafts/",
        fig_name="confidence_regions_bimodal_500.pdf", save=False):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

    # ax.figure(figsize=(figsize * 3, figsize))
    # ax.subplot(131)
    ax.plot(bimodal_data[0][:, 0], bimodal_data[0][:, 1], 'k.', alpha=0.3)
    ax.plot(2, 2, 'kx', mew=2, ms=10, label='Mean of dominant Gaussian')
    ax.plot(0, 0, 'x', color='grey',
            mew=2, ms=10, label='Mean of subdominant Gaussian')
    ax.legend(loc='best', frameon=False)

    if save:
        ax.savefig(fig_name, bbox_inches='tight')

    return


def plot_dumbbell_500_comparison(
        dumbbell_data2,
        shrink_peak_dens2,
        KDE_peak_dens2,
        KDE_peak_dens2b,
        cent_peak_dens2,
        figsidesize=7,
        plot_path="../../paper/figures/drafts/",
        plot_fig_name="confidence_regions_dumbbell_500.pdf", ax=None):
    """ for plot in appendix, tweaked labels to look best on fig"""

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

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
                          figsize=7, save=False, ax=None):

    ax.figure(figsize=(figsize * 3, figsize))
    ax.subplot(131)
    ax.plot(gauss_data[0][:, 0], gauss_data[0][:, 1], 'k.', alpha=0.3)
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

plot_error_as_a_func_of_data_pts.__doc__ = \
    """
    :param f: hdf5 hierarchical structure


    f
    |
    |_20__"gauss"_____"KDE"
    |      |          |
    |      |          "shrink"
    |      |          |
    |      |          "cent"
    |      |
    |      "bimodal"__"KDE"
    |      |          |
    |      |          "shrink
    |      |          |
    |      |          "cent"
    |      |
    |      |
    |      "dumb"_____"KDE"
    |                 |
    |                 "shrink
    |                 |
    |                 "cent"
    |
    |_50 (with children structure the same as `20`)
    |_100
    |_500

    """
