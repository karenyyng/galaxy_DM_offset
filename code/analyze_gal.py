"""analyze galaxy physical properties from Illustris-1 simulation
Author: Karen Ng <karenyng@ucdavis.edu>
License: BSD
"""
import matplotlib.pyplot as plt
import numpy as np
import rpy2.robjects as robjects


def make_color_mag_diag(df, bluer_band, redder_band, band_limit,
                        plot=False, phot_band=None, save=False,
                        savePath="../plots/", clst=None):
    """
    :parameters:
    df = pandas df of each cluster
    bluer_band = string, df colname
    red_band = string, df colname
    band_limit = float,
        how many band magnitude fainter than BCG do we want to examine

    :returns: None

    :stability: works
    """
    if phot_band is not None:
        df.rename(columns=phot_band, inplace=True)
    bcg_i = df[redder_band].min()
    mask_i = df[redder_band] < bcg_i + band_limit
    g_i = df[bluer_band][mask_i] - df[redder_band][mask_i]

    plt.plot(df[redder_band][mask_i], g_i, "ro", fillstyle="none")
    plt.title("Color-magnitude diagram for" +
              " {0} subhalos".format(np.sum(mask_i)))
    plt.ylabel(bluer_band + " - " + redder_band)
    plt.xlabel(redder_band)
    if save is True:
        assert clst is not None, "arg for clst missing"
        plt.savefig(savePath + "cm_diagram{0}.png".format(clst),
                    bbox_inches="tight")
    plt.show()
    return


def rKDE():
    """do KDE by calling R package
    return density estimate
    """

    return


def weighted_centroid():
    """
    """
    return
