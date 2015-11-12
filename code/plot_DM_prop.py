from __future__ import (division, print_function)
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../")
import compute_distance as get_dist


def plot_DM_fhat(fhat, fhat_stars, clstNo, threshold=0.3,
                 convert_kpc_over_h_to_kpc=True, fontsize=25):

    if "log_est" not in fhat.keys():
        fhat["log_est"] = np.log(fhat["estimate"])

    plt.subplot('111', axisbg='black', aspect='equal')

    # Plot DM particle histograms
    plt.contourf(fhat["eval_points"][0], fhat["eval_points"][1],
                 fhat["log_est"].transpose(), cmap=plt.cm.afmhot)

    ((dist, matched_DM_ixes), sign_gal_peak_no, sign_DM_peak_no) = \
        get_dist.compute_distance_between_DM_and_gal_peaks(fhat_stars, fhat)

    # Plot DM peaks
    # Peaks that are associated with the galaxy peaks are in blue.
    # Peaks that are not associated with the galaxy peaks are in cyan.
    plt.plot(fhat["peaks_xcoords"][:sign_DM_peak_no],
             fhat["peaks_ycoords"][:sign_DM_peak_no],
             "o", color='cyan', fillstyle="none", mew=3,
             ms=35, label="candidate DM peaks")

    plt.plot(fhat["peaks_xcoords"][matched_DM_ixes],
             fhat["peaks_ycoords"][matched_DM_ixes],
             "o", color='blue', fillstyle="none", mew=3,
             ms=35, label="matched DM peaks")

    # Plot I-band luminosity peaks
    # Peaks with density > 0.5 density of densest peak are in red.
    # Peaks with density < 0.5 density of densest peak are in pink.
    if convert_kpc_over_h_to_kpc:
        unit_conversion = 1. / .704
        print("Converting unit of kpc / h to kpc for galaxy data using ")
        print (unit_conversion)
        plt.plot(fhat_stars["peaks_xcoords"][:sign_gal_peak_no] * unit_conversion,
                 fhat_stars["peaks_ycoords"][:sign_gal_peak_no] * unit_conversion,
                 'o', color='m', fillstyle="none", mew=3, ms=35,
                 label="significant I band luminosity peaks")

    offset_string = ["{0:0.0f}".format(i) for i in dist]
    offset_string = ', '.join(offset_string)
    plt.title('Cluster {0}: offset(s) = {1} kpc'.format(clstNo, offset_string),
              size=fontsize)

    # Make ticks bigger
    plt.tick_params(axis='both', which='both', labelsize=fontsize)

    lgd = plt.legend(loc='best', fontsize=fontsize, frameon=1)
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    plt.xlabel('kpc', size=fontsize)
    plt.ylabel('kpc', size=fontsize)

    return



