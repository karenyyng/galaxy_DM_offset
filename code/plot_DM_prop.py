from __future__ import (division, print_function)
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../")
import compute_distance as get_dist


def plot_DM_fhat(fhat, fhat_stars, clstNo, threshold=0.3,
                 convert_kpc_over_h_to_kpc=True, fontsize=25):
    peaks_mask = fhat["peaks_dens"] > threshold

    if "log_est" not in fhat.keys():
        fhat["log_est"] = np.log(fhat["estimate"])

    plt.subplot('111', axisbg='black', aspect='equal')

    # Plot DM particle histograms
    plt.contourf(fhat["eval_points"][0], fhat["eval_points"][1],
                 fhat["log_est"].transpose(), cmap=plt.cm.afmhot)

    ixes = np.arange(0, len(fhat["peaks_xcoords"]), 1)[peaks_mask]

    get_dist.compute_distance_between_DM_and_gal_peaks(fhat_stars, fhat)

    # Plot DM peaks
    # Peaks that are associated with the galaxy peaks are in blue.
    # Peaks that are not associated with the galaxy peaks are in cyan.
    plt.plot(fhat["peaks_xcoords"][ixes], fhat["peaks_ycoords"][ixes],
             "o", color='cyan', fillstyle="none", mew=3,
             ms=35, label="DM peaks")

    # Plot I-band luminosity peaks
    # Peaks with density > 0.5 density of densest peak are in red.
    # Peaks with density < 0.5 density of densest peak are in pink.
    if convert_kpc_over_h_to_kpc:
        print("Converting unit of kpc / h to kpc for galaxy data")
        plt.plot(fhat_stars["peaks_xcoords"] * 106.5 / 75.,
                 fhat_stars["peaks_ycoords"] * 106.5 / 75.,
                 'o', color='red', fillstyle="none", mew=3, ms=35,
                 label="I band luminosity peaks")

    plt.title('Cluster {0}: DM peak density threshold = {1}'.format(clstNo,
                                                                    threshold),
              size=fontsize)

    # Make ticks bigger
    plt.tick_params(axis='both', which='both', labelsize=fontsize)

    lgd = plt.legend(loc='best', fontsize=fontsize, frameon=1)
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    plt.xlabel('kpc', size=fontsize)
    plt.ylabel('kpc', size=fontsize)

    return



