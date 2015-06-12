from __future__ import (division, print_function)
import matplotlib.pyplot as plt
import numpy as np


def plot_DM_fhat(fhat, fhat_stars, clstNo, threshold=0.3,
                 convert_kpc_over_h_to_kpc=True):
    peaks_mask = fhat["peaks_dens"] > threshold

    if "log_est" not in fhat.keys():
        fhat["log_est"] = np.log(fhat["estimate"])

    plt.subplot('111', axisbg='black', aspect='equal')

    # plot DM particle histograms
    plt.contourf(fhat["eval_points"][0], fhat["eval_points"][1],
                 fhat["log_est"].transpose(), cmap=plt.cm.afmhot)

    ixes = np.arange(0, len(fhat["peaks_xcoords"]), 1)[peaks_mask]
    # plot DM peaks
    plt.plot(fhat["peaks_xcoords"][ixes], fhat["peaks_ycoords"][ixes],
             "o", color='cyan', fillstyle="none", mew=3,
             ms=35, label="DM peaks")

    # plot I-band luminosity peaks
    if convert_kpc_over_h_to_kpc:
        print("Converting unit of kpc / h to kpc for galaxy data")
        plt.plot(fhat_stars["peaks_xcoords"] * 106.5 / 75.,
                fhat_stars["peaks_ycoords"] * 106.5 / 75.,
                'o', color='red', fillstyle="none", mew=3, ms=35,
                label="I band luminosity peaks")

    plt.title('Cluster {0}: DM peak density threshold = {1}'.format(clstNo,
                                                                    threshold),
              size=30)

    # make ticks bigger
    plt.tick_params(axis='both', which='both', labelsize=30)

    lgd = plt.legend(loc='best', fontsize=30, frameon=1)
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    plt.xlabel('kpc', size=30)
    plt.ylabel('kpc', size=30)

    return
