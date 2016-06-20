from __future__ import (division, print_function)
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../")
import compute_distance as get_dist


def plot_DM_fhat(fhat, fhat_stars, clstNo, threshold=0.3,
                 convert_kpc_over_h_to_kpc=True, fontsize=13,
                 unit_conversion = 1. / .704, ax=None, markersize=25,
                 log_scale=True, legend_box_anchor=(1.0, 1.2),
                 legend_markerscale=0.7, kernel_width=1, flip_y=-1.,
                 xlabel_rotate_angle=45, verbose=False, origin='lower',
                 show_legend=True, power=1./20.):
    """
    :param kernel_width: float, the number times the histogram size = 2 kpc
    """

    rc("font", family="serif")
    if log_scale and "log_est" not in fhat.keys():
        log_est = np.log(fhat["estimate"][:])
    else:
        log_est = np.power(fhat["estimate"][:], power)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot('111', axisbg='black', aspect='equal')

    if type(fhat) == dict:
        extent = [
                    fhat["eval_points"][0][0], fhat["eval_points"][0][-1],
                    fhat["eval_points"][1][0], fhat["eval_points"][1][-1],
                 ]
    else:
        extent = [
                    fhat["eval_points0"][0], fhat["eval_points0"][-1],
                    fhat["eval_points1"][0], fhat["eval_points1"][-1],
                 ]
    # Plot DM particle histograms
    ax.imshow(log_est.transpose(), cmap=plt.cm.afmhot,
               extent=extent, origin=origin
               )
    # ax.imshow(log_est.transpose(), cmap=plt.cm.afmhot,
    #            extent=extent, origin='upper'
    #            )

    matched_dict = \
        get_dist.compute_distance_between_DM_and_gal_peaks(fhat_stars, fhat,
                                                           verbose=verbose)

    dist = matched_dict["dist"]
    matched_DM_ixes = matched_dict["DM_ixes"]
    sign_gal_peak_no = matched_dict["gal_peak_no"]
    sign_DM_peak_no = matched_dict["DM_peak_no"]
    # gd_threshold = matched_dict["gd_threshold"]

    # Plot DM peaks
    # Peaks that are associated with the galaxy peaks are in blue.
    # Peaks that are not associated with the galaxy peaks are in cyan.
    ax.plot(fhat["peaks_xcoords"][:sign_DM_peak_no],
            fhat["peaks_ycoords"][:sign_DM_peak_no],
            "o", color='cyan', fillstyle="none", mew=3,
            ms=markersize, label="candidate DM peaks", alpha=0.5)


    # Plot I-band luminosity peaks
    # Peaks with density > 0.5 density of densest peak are in red.
    # Peaks with density < 0.5 density of densest peak are in pink.
    if convert_kpc_over_h_to_kpc:
        if verbose:
            print("Converting unit of kpc / h to kpc for galaxy data using ")
            print (unit_conversion)
    else:
        unit_conversion = 1.
    ax.plot(
        fhat_stars["peaks_xcoords"][:sign_gal_peak_no] * unit_conversion,
        fhat_stars["peaks_ycoords"][:sign_gal_peak_no] * unit_conversion,
        'o', color='m', fillstyle="none", mew=3, ms=markersize,
        label="significant I band luminosity peaks")

    ax.plot(fhat["peaks_xcoords"][:][matched_DM_ixes],
            fhat["peaks_ycoords"][:][matched_DM_ixes],
             "o", color='blue', fillstyle="none", mew=3,
             ms=markersize, label="matched DM peaks", alpha=0.8)

    offset_string = ["{0:0.0f}".format(i) for i in dist]
    offset_string = ', '.join(offset_string)
    ax.set_title('Cluster {0}: gal-DM offset(s) = {1} kpc'.format(
        clstNo, offset_string) +
        ", \n kernel size= {0} kpc".format(kernel_width * 2.)
        , size=fontsize*1.2)

    # Make ticks bigger
    ax.tick_params(axis='both', which='both', labelsize=fontsize)
    # xtickslabels = ax.get_xticklabels()
    # ax.set_xticklabels(xtickslabels, rotation=45)

    if show_legend:
        lgd = ax.legend(fontsize=int(fontsize * 1.1), frameon=1,
                        numpoints=1, bbox_to_anchor=legend_box_anchor,
                        loc='upper right', markerscale=legend_markerscale)
        frame = lgd.get_frame()
        frame.set_facecolor('white')
    ax.set_xlabel('kpc', size=fontsize)
    ax.set_ylabel('kpc', size=fontsize)

    # flip back the axes
    ylabels = [label * flip_y for label in ax.get_yticks().tolist()]
    ax.set_yticklabels(ylabels)

    # plot the smoothing scale
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    lower_x_bar = xlims[1] - np.diff(xlims) * 0.2
    y_bar_location = ylims[0] + np.diff(ylims) * 0.20
    ax.plot((lower_x_bar, lower_x_bar + kernel_width * 2),
            (y_bar_location, y_bar_location), 'w', linewidth=5)

    # rotate tick label
    map(lambda x: x.set_rotation(xlabel_rotate_angle),
        ax.get_xticklabels())

    if verbose:
        print ("gal peak = ({0:0.0f}, {1:0.0f})".format(
                fhat_stars["peaks_xcoords"][:sign_gal_peak_no][0] * unit_conversion,
                fhat_stars["peaks_ycoords"][:sign_gal_peak_no][0] * unit_conversion,
               ))

        print ( "The matched DM peak = ({0:0.0f}, {1:0.0f})".format(
            fhat["peaks_xcoords"][:][matched_DM_ixes][0],
            fhat["peaks_ycoords"][:][matched_DM_ixes][0],
            )
        )

    return ax, xlims, ylims



