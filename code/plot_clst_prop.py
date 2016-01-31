from __future__ import division
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
sys.path.append("../")
import get_gal_centroids as get_gal
import calculate_astrophy_quantities as cal_astrophy


def plot_cluster_mass_distribution(groupMass, groupMcrit200 , groupMcrit500,
                                   y_legend, x_label, y_label, ax=None,
                                   save=True, path="../../paper/figures/",
                                   ticksize=12, labelsize=15):
    """
    :param ticks: numpy array of floats
    :param y_data: list of numpy array of data
    :param x_label: str, x axis label of the plot
    :param y_label: str, y axis label of the plot
    :param ax: matplotlib ax object
    :param save: bool, whether to save the plot to file
    :param path: str, path of the directory to save the file

    :return ax:
    """
    ticks, countGroupMass = compute_clst_no_above_mass_threshold(groupMass)
    ticks, countGroupMcrit200 = \
        compute_clst_no_above_mass_threshold(groupMcrit200)
    ticks, countGroupMcrit500 = \
        compute_clst_no_above_mass_threshold(groupMcrit500)

    y_data = [countGroupMass, countGroupMcrit200, countGroupMcrit500]
    y_legend = [r"$M_{\rm FoF}$", r"$M_{200c}$", r"$M_{500c}$"]
    x_label = r"$M_{Cluster}(M_{\odot})$"
    y_label = r"$N(> M_{Cluster})$"

    assert len(y_legend) == len(y_data), "number of legends has to match " + \
        " length of data"

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

    ticks = np.vstack((ticks, ticks)).transpose().ravel()[1:]
    ticks = np.concatenate((ticks, np.array([1e5])))
    ticks *= 1e10

    for i in range(len(y_data)):
        y = y_data[i]
        y = np.array(y)
        y = np.vstack((y, y)).transpose().ravel()
        ax.loglog(ticks, y, lw=2, label=y_legend[i])

    ax.set_title(
        r"Mass distribution of Illustris clusters" + "\n" +
        r"above certain mass",
        fontsize=12)
    ax.set_xlabel(x_label, fontsize=labelsize)
    ax.set_ylabel(y_label, fontsize=labelsize)
    ax.tick_params(width=1.5, length=8, which='major', labelsize=ticksize)
    ax.tick_params(width=1.5, length=6, which='minor', labelsize=ticksize)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.legend(loc='best', fontsize=labelsize)

    if save:
        filename = path + "clusterMassDist.eps"
        print "saving figure {0}".format(filename)
        plt.savefig(filename, bbox_inches="tight")

    return ax


def compute_clst_no_above_mass_threshold(mass,
                                         thresHoldRange=np.array([1e13, 1e14])
                                         / 1e10):
    """
    :param mass: np.array in units of 1e10 solar mass
    :param thresHoldRange: numpy array in units of 1e10 solar mass

    :return ticks: numpy array with the ticks of cumlative plot
    :return noCount: numpy array with number of data point in each tick bin
    """
    ticks1 = np.arange(thresHoldRange[0], 10 * thresHoldRange[0],
                       step=thresHoldRange[0])
    ticks2 = np.arange(thresHoldRange[1], 10 * thresHoldRange[1],
                       step=thresHoldRange[1])
    ticks = np.concatenate([ticks1, ticks2])

    noCount = [np.sum(mass > t) for t in ticks]

    return ticks, noCount


def visualize_3D_clst(df, position_keys):
    """
    :param df: pandas data frame with the position keys
    :param position_keys: tuple of strings that correspond to the keys in df
        for the position columns
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('x (Mpc/h)')
    ax.set_ylabel('y (Mpc/h)')
    ax.set_zlabel('z (Mpc/h)')

    ax.plot(df[position_keys[0]], df[position_keys[1]], df[position_keys[2]],
            'o', alpha=0.05, label='subhalo location')

    ax.legend(loc='best')

    plt.show()


def plot_mass_vs_richness(FoF_mass, clst_df_list, ax=None,
                          z_ranges=(0.3, 0.4, 0.5), show=False,
                          prop_cycler=None):
    """FoF mass vs richness assuming
    :FoF_mass: numpy array of floats, each float corresponds to the mass of each
    cluster
    :clst_df_list: list of pandas dataframes, each df contains the subhalos

    """
    from cycler import cycler
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    Illustris_cosmology = cal_astrophy.get_Illustris_cosmology()

    # Find limiting magnitude for each assumed cosmological z
    limiting_mag_list = map(lambda z:
        cal_astrophy.dimmest_mag_for_illustris_halos_to_be_observable(
        astropy_cosmo=Illustris_cosmology, z=z),
        z_ranges
    )
    if prop_cycler is not None:
        ax.set_prop_cycle(prop_cycler)

    mask = {}
    richness = {}

    # Find how many subhalos survives after the cut for each cluster
    # for each redshift
    for i, z in enumerate(z_ranges):

        mask[z] = map(lambda df:
                      get_gal.cut_reliable_galaxies(
                        df, limiting_mag=limiting_mag_list[i]),
                      clst_df_list
                      )

        richness[z] = map(lambda m: np.sum(m), mask[z])

    for z in z_ranges[::-1]:
        ax.plot(FoF_mass, richness[z], 'o', label="assumed z = {}".format(z),
                mew=1.5, fillstyle='none', alpha=0.8)
        ax.legend(loc='upper left', frameon=True)
        ax.set_xscale("log", nonposx='clip')
        ax.set_xlim(1e13, 1e15)

    ax.set_xlabel(r"$M_{FoF}$")
    ax.set_ylabel("Richness")

    # richness cut
    # ax.axhline(50, color='k')
    # ax.text(1.8e14, 55,"richness cut > 50", size=12)
    # ax.arrow(1.6e14, 50, 0, 30, head_width=1e13, head_length=20,
    #          fc='k', ec='k')


    if not show:
        plt.close()

    return richness, mask


def N_by_N_lower_triangle_plot(data, space, var_list, axlims=None,
                               Nbins_2D=None, axlabels=None, N_bins=None,
                               xlabel_to_rot=None, histran=None, figsize=6,
                               fontsize=12, save=False, prefix=None,
                               suffix=".png", path="./"):
    """ create a N by N matrix of plots
    with the top plot of each row showing a density plot in 1D
    and the remaining plots being 2D contour plots
    df = dataframe that contain the data of all the variables to be plots
    space = float, px of space that is added between subplots
    var_list = list of strings - denotes the column header names
        that needs to be plotted
    axlims = dictionary, keys are the strings in var_list,
        each value is a tuple of (low_lim, up_lim) to denote the limit
        of values to be plotted
    Nbins_2D = dictionary, keys are in format of tuples of
        (x_col_str, y_col_str) to denote which subplot you are referring to
    axlabels = dictionary, keys correspond to the variable names
    xlabel_to_rot = dictionary,
        key is the the key for the labels to be rotated,
        value is the degree to be rotated
    histran = dictionary,
        some keys has to be the ones for the plots, value are in
        form of (lowerhist_range, upperhist_range)
    figsize = integer, figuares are squared this refers to the side length
    fontsize = integer, denotes font size of the labels
    save = logical, denotes if plot should be saved or not
    prefix = string, prefix of the output plot file
    path = string, path of the output plot file
    suffix = string, file extension of the output plot file

    Stability: Not entirely tested, use at own risk
    """
    from matplotlib.ticker import MaxNLocator

    # begin checking if inputs make sense
    N = len(var_list)
    assert N <= len(axlabels), "length of axlabels is wrong"
    assert N >= 2, "lower triangular contour plots require more than 2\
        variables in the data"

    for var in var_list:
        assert var in data.columns, "variable to be plotted not in df"

    if axlabels is None:
        axlabels = {key: key for key in var_list}

    if xlabel_to_rot is None:
        xlabel_to_rot = {key: 0 for key in var_list}

    if histran is None:
        histran = {key: None for key in var_list}

    if axlims is None:
        axlims = {key: (None, None) for key in var_list}

    if Nbins_2D is None:
        keys = comb_zip(var_list, var_list)
        Nbins_2D = {key: 50 for key in keys}

    if N_bins is None:
        N_bins = {key: 'knuth' for key in var_list}

    if save:
        assert prefix is not None, "prefix for output file cannot be none"

    # impossible for the matrix plot not to be squared in terms of dimensions
    # set each of the subplot to be squared with the figsize option
    f, axarr = pylab.subplots(N, N, figsize=(figsize, figsize))
    f.subplots_adjust(wspace=space, hspace=space)

    # remove unwanted plots on the upper right
    plt.setp([a.get_axes() for i in range(N - 1)
              for a in axarr[i, i + 1:]], visible=False)

    # remove unwanted row axes tick labels
    plt.setp([a.get_xticklabels() for i in range(N - 1)
              for a in axarr[i, :]], visible=False)

    # remove unwanted column axes tick labels
    plt.setp([axarr[0, 0].get_yticklabels()], visible=False)
    plt.setp([a.get_yticklabels() for i in range(N - 1)
              for a in axarr[i + 1, 1:]], visible=False)

    # create axes labels
    if axlabels is not None:
        for j in range(1, N):
            axarr[j, 0].set_ylabel(axlabels[var_list[j]], fontsize=fontsize)
        for i in range(N):
            axarr[N - 1, i].set_xlabel(axlabels[var_list[i]],
                                       fontsize=fontsize)

    for n in range(N):
        # avoid overlapping lowest and highest ticks mark
        # print "setting x and y tick freq for {0}".format((n, n))
        ax2 = axarr[n, n]
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))

    # print "setting x and y tick freq for {0}".format((i, j))
    for i in range(N):
        for j in range(N):  # range(i)
            ax2 = axarr[i, j]
            ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
            ax2.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))

    # rotate the xlabels appropriately
    if xlabel_to_rot is not None:
        match_ix = [var_list.index(item) for item in var_list]
        # ok to use for-loops for small number of iterations
        for ix in match_ix:
            labels = axarr[N - 1, ix].get_xticklabels()
            for label in labels:
                label.set_rotation(xlabel_to_rot[var_list[ix]])

    # start plotting the diagonal
    for i in range(N):
        print "N_bins = {0}".format(N_bins[var_list[i]])
        histplot1d_part(axarr[i, i], np.array(data[var_list[i]]),
                        np.array(data['prob']),
                        N_bins=N_bins[var_list[i]],
                        histrange=histran[var_list[i]],
                        x_lim=axlims[var_list[i]])

    # start plotting the lower triangle when row no > col no
    for i in range(N):
        for j in range(i):
            histplot2d_part(axarr[i, j], np.array(data[var_list[j]]),
                            np.array(data[var_list[i]]),
                            prob=np.array(data['prob']),
                            N_bins=Nbins_2D[(var_list[j], var_list[i])],
                            x_lim=axlims[var_list[j]],
                            y_lim=axlims[var_list[i]])

    if save:
        print "saving plot to {0}".format(path + prefix + suffix)
        plt.savefig(path + prefix + suffix, dpi=200, bbox_inches='tight')

    return



