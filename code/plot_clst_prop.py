from __future__ import division
import matplotlib.pyplot as plt
import numpy as np


def plot_cluster_mass_distribution(ticks, y_data,
                                   y_legend, x_ticks, y_ticks,
                                   save=True, path="../../paper/figures/"):
    """
    :param ticks: numpy array of floats
    :y_data: list of numpy array of data

    """
    assert len(y_legend) == len(y_data), "number of legends has to match " + \
        " length of data"

    ticks = np.vstack((ticks, ticks)).transpose().ravel()[1:]
    ticks = np.concatenate((ticks, np.array([1e5])))

    for i in range(len(y_data)):
        y = y_data[i]
        y = np.array(y)
        y = np.vstack((y, y)).transpose().ravel()
        plt.loglog(ticks, y, lw=2, label=y_legend[i])

    plt.title("Illustris clusters at z=0")
    plt.xlabel(x_ticks)
    plt.ylabel(y_ticks)
    plt.legend(loc='best')

    if save:
        filename = path + "clusterMassDist.eps"
        print "saving figure {0}".format(filename)
        plt.savefig(filename, bbox_inches="tight")


    return


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
