from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_cluster_mass_distribution(ticks, y_data,
                                   y_legend, x_label, y_label, ax=None,
                                   save=True, path="../../paper/figures/",
                                   labelsize=10):
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
        r"Mass distribution of Illustris clusters at $z=0$ above certain mass",
        fontsize=12)
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.tick_params(width=1.5, length=8, which='major', labelsize=labelsize)
    ax.tick_params(width=1.5, length=6, which='minor', labelsize=labelsize)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.legend(loc='best')

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


def plot_mass_vs_richness(df):
    """TODO: Docstring for plot_mass_vs_richness.
    :returns: TODO

    """
    pass
