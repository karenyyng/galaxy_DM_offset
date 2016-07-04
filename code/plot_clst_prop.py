from __future__ import division
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
sys.path.append("../")
import get_gal_centroids as get_gal
import calculate_astrophy_quantities as cal_astrophy
import plot_cred_int as plotCI
import matplotlib.gridspec as gridspec
from astropy.stats import biweight_location



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
                          prop_cycler=None, show_richness_cut=False,
                          richness_cut=50):
    """FoF mass vs richness assuming
    :FoF_mass: numpy array of floats, each float corresponds to the mass of each
    cluster
    :clst_df_list: list of pandas dataframes, each df contains the subhalos
    :show_richness_cut: bool
    :richness_cut: integer, what is the minimum number of subhalos for a
        cluster to be considered
    """
    # from cycler import cycler
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
                      get_gal.cut_dim_galaxies(
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
    if show_richness_cut:
        ax.axhline(50, color='k')
        ax.text(1.8e14, 55,"richness cut > 50", size=12)
        ax.arrow(1.6e14, 50, 0, 30, head_width=1e13, head_length=20,
                 fc='k', ec='k')


    if not show:
        plt.close()

    return richness, mask


def plot_2D_offsets(df, stat_key, prefices=["Delta_x_", "Delta_y_"],
                    suffices=["_x", "_y"], prefix_bool=1):
    for i, key in enumerate(stat_key) :
        plt.axes().set_aspect('equal')
        if prefix_bool:
            fixed_keys = [prefices[0] + key, prefices[1] + key]
        else:
            fixed_keys = [key + suffices[0], key + suffices[1]]

        plt.plot(
            np.array(df[fixed_keys[0]]),
            np.array(df[fixed_keys[1]]), 'b.', alpha=0.05
            )
        biweight_loc = (
            biweight_location(df[fixed_keys[0]]),
            biweight_location(df[fixed_keys[1]]))

        # The red cross is the biweight location along each dimension
        plt.plot(biweight_loc[0], biweight_loc[1],
             'rx', mew=2.)
        plt.tick_params(labeltop='off', labelright='off')
        plt.axes().yaxis.set_ticks_position('left')
        plt.axes().xaxis.set_ticks_position('bottom')
        plt.xlim(-300, 300)
        plt.ylim(-300, 300)
        plt.title(key + ', biweight_loc = {0:.2f}, {1:.2f}'.format(
            *biweight_loc))

        plt.show()
        plt.clf()


def plot_offset_along_1_dimension(
        df, stat_key_dict, save=False, lvls=[68., 95., 99.],
        savefile=None, xlims=(-400.,400.)):
    key_length = len(stat_key_dict)
    sum_stat_df_dict = {}
    fig = plt.figure(figsize=(18, 3 * key_length))
    gs = gridspec.GridSpec(key_length, 1)
    gs.update(hspace=0.4)
    gs.set_width_ratios([1., 1.])


    ax_lists = [[fig.add_subplot(gs[row, col]) for col in range(1)]
            for row in range(key_length)]

    for i, stat in enumerate(stat_key_dict):
        ax = ax_lists[i][0]
        sum_stat_df_dict[stat] = \
            plotCI.CI_loc_plot(
                np.array(df[stat]), ax=ax, lvls=lvls)
        ax.set_xlim(*xlims)
        ax.set_xlabel(stat_key_dict[stat] + ' (kpc)')
        ax.set_ylabel('PDF')
        ax.tick_params(labeltop='off', labelright='off')
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    if save and savefile is not None:
        fig.savefig(savefile, bbox_inches='tight')
    elif save and savefile is None:
        raise ValueError("`savefile` cannot be None.")

    return sum_stat_df_dict
