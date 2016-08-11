"""plot data properties in gal clusters from Illustris-1 simulation
Author: Karen Ng <karenyng@ucdavis.edu>
License: BSD
"""
from __future__ import division, print_function
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
# from matplotlib.mlab import bivariate_normal
import numpy as np
# import rpy2.robjects as robjects

# my own modules
import calculate_astrophy_quantities as cal_astro
from matplotlib import rc
rc("font", family="serif")


def plot_color_mag_diag(df, bluer_band, redder_band, band_limit,
                        title_mag_limit=24.4,
                        plot=False, save=False, subhalo_len_lim=1e3,
                        savePath="../plots/", clst=None, verbose=True,
                        convert_to_apparent_mag=True, assume_z=0.3,
                        highlight_observable_subhalos=True,
                        fileprefix="color_magnitude_diagram_clst",
                        closePlot=False, showPlot=True
                        ):
    """
    :parameters:
    df = pandas df of each cluster
    bluer_band = string, df colname
    red_band = string, df colname
    band_limit = float,
        how many band magnitude fainter than BCG do we want to show
    title_mag_limit = float,
        the band limit below which we highlight because subhalos like those
        should be observable
    particleLim = int,
        want to ensure subhalos has at least that many DM `particles`
    savePath = string, file path directory to save to
    clst = string, string or integer that denotes the ID of the cluster that is
        visualized
    verbose = bool, whether to show the plot or not
    convert_to_apparent_mag = bool, whether to convert the absolute magitude to
    apparent magnitude
    assume_z = float, the cosmological redshift assumed when we converting
    absolute magnitude to apparent magnitude

    :returns: None

    :stability: works
    """
    # compute the color first
    bcg_i = df[redder_band].min()
    mask_i = df[redder_band] < bcg_i + band_limit
    g_i = df[bluer_band] - df[redder_band]

    if convert_to_apparent_mag and verbose:
        print ("Converting apparent magnitude to absolute magnitude\n")
        print ("assuming the cosmological redshift is z = {}".format(assume_z))

    if convert_to_apparent_mag:
        # convert the magnitude to absolute magnitude
        Illustris_cosmo = cal_astro.get_Illustris_cosmology()
        df['apparent_' + redder_band] = \
            df[redder_band].apply(
                lambda x:
                cal_astro.convert_abs_mag_to_apparent_mag(x, Illustris_cosmo,
                                                          z=assume_z))

    # if verbose:
    #     print("subhalos need at least {0} DM".format(subhalo_len_lim) +
    #           " particles to be plotted")

    # examine number of DM particles
    mask_ii = df["SubhaloLenType1"] > subhalo_len_lim
    mask_i = np.logical_and(mask_i, mask_ii)
    observable_mask = df['apparent_' + redder_band] < title_mag_limit

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.tick_params(labeltop='off', labelright='off')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    if convert_to_apparent_mag:
        ax.plot(df['apparent_' + redder_band][mask_i],
                 g_i[mask_i], "bo", fillstyle='none')
        # ax.plot(df['apparent_' + redder_band][observable_mask],
        #          g_i[observable_mask], "bo", fillstyle='none',
        #         alpha=0.5)
    else:
        ax.plot(df[redder_band][mask_i], g_i[mask_i], "k.", alpha=0.3)

    label_bluer_band = bluer_band.replace("_", "-")
    label_redder_band = redder_band.replace("_", "-")
    ax.set_ylabel(label_bluer_band + " - " + label_redder_band)
    ax.set_xlabel(label_redder_band)

    ax.set_ylim(0, 0.5)

    if clst is not None and not highlight_observable_subhalos:
        ax.set_title("Cluster {0}: Color-magnitude diagram with".format(clst))

    elif clst is not None and highlight_observable_subhalos:
        ax.set_title("Cluster {0} ".format(clst) +
                  r"with {0} subhalos with ".format(np.sum(observable_mask)) +
                     "\napparent $i$ < {0}".format(title_mag_limit) +
                  " assuming cosmological z = {}".format(assume_z),
                     fontsize=14
                  )

    if save is True:
        assert clst is not None, "arg for clst missing"
        plt.savefig(savePath + "/" + fileprefix + "{0}.eps".format(clst),
                    bbox_inches="tight")

    if showPlot:
        plt.show()

    if closePlot:
        plt.close()

    return


def plot_cf_contour(dens, x, y, lvls=[68, 95], clabel=False,
                    fill=False, colors=None, ax=None):
    """this sort through the density, add them up til they are
    below the required confidence level, then
    draw contour lines at the values specified in sequence lvl_vals

    :param dens: np.array, the density estimate, should integrate to 1
    :param x: np.array, x coord of the density estimate
    :param y: np.array, y coord of the density estimate
    :param lvls: list of floats, denotes percentile, 65 means 65th percentile
    :param show: boolean, whether to show the plot
    :param clabel: boolean, whether to add contour label indicating the level
    :param fill: boolean, whether to fill the contour
    :param colors: list of tuples,
        each tuple should contain 3 color float values to use for contours
        the list should be as long as `lvls`

    :returns: dictionary containing points that correspond to the contours
        key of the dictionary are the `lvls` supplied
    :notes:
    http://stackoverflow.com/questions/5666056/matplotlib-extracting-data-from-contour-lines
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    d = dens.ravel()
    lvls = np.array(lvls) / 100.
    lvl_vals = np.zeros(len(lvls))
    sums = 0

    d = np.sort(d)  # in ascending order
    d_sum = np.sum(d)  # compute normalization

    for j in xrange(d.size):
        sums += d[j]
        # compute the different confidence levels
        for i in range(len(lvls)):
            if sums / d_sum <= 1. - lvls[i]:
                lvl_vals[i] = d[j]

    if colors is None:
        colors = \
            [((len(lvls) - i) / len(lvls), 0 / 255., 100 / 255.)
             for i in range(len(lvls))]

    # the plt.contour function is weird, if you don't transpose
    # the density, the plotted density will be rotated by 180 clockwise
    if fill is False:
        CS = ax.contour(x, y, dens.transpose(), lvl_vals, linewidths=(1, 1),
                        colors=colors)
    else:
        CS = ax.pcolor(x, y, dens.transpose(), cmap=plt.cm.winter)

    if clabel:
        # labels for the confidence levels
        str_lvls = {l: "{0:.1f}".format(s * 100)
                    for l, s in zip(CS.levels, lvls)}
        ax.clabel(CS, CS.levels, fmt=str_lvls, inline=1, fontsize=6.5)

    lvl_contours = {}
    for i, lvl in enumerate(lvls):
        p = CS.collections[i].get_paths()[0]
        lvl_contours[lvl] = p.vertices
    return lvl_contours


def plot_KDE_peaks(fhat, lvls=range(2, 100, 10), allPeaks=True,
                   plotDataPoints=False, save=False, R200C=None,
                   fileName="KDE_plot_cluster", clstNo=None,
                   clabel=False, showData=False, xlabel="x (kpc / h)",
                   ylabel="y (kpc / h)", showDomPeak=True,
                   fileDir="../plots/", fill=False, showContour=True,
                   ax=None, fig=None, xlabel_rotate_angle=90,
                   legend_box_anchor=(1., 1.4),
                   convert_kpc_over_h_to_kpc=False, flip_y=1.,
                   legend_markerscale=0.5, unit_conversion=1./0.704,
                   xlims=None, ylims=None, colorbar_ax=None
                   ):
    """make a plot of the fhat along with other important info
    :param fhat: dictionary or hdf5 filestream,
        that is generated from `do_KDE_and_get_peaks`
    :returns: color bar instance
    """
    if not convert_kpc_over_h_to_kpc:
        unit_conversion = 1.
    else:
        xlabel = "kpc"
        ylabel = "kpc"

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

    # flip histogram to match the DM contours
    if showContour:
        plot_cf_contour(fhat["estimate"][:],
                        fhat["eval_points"][0] * unit_conversion,
                        fhat["eval_points"][1] * flip_y * unit_conversion,
                        lvls=lvls, clabel=clabel, fill=fill, ax=ax)

    if plotDataPoints:
        ax.plot(fhat["data_x"].transpose()[0] * unit_conversion,
                fhat["data_x"].transpose()[1] * unit_conversion * flip_y,
                'r.', alpha=1)

    if 'BCG' in fhat:
        ax.plot(fhat['BCG'][0] * unit_conversion,
                flip_y * fhat['BCG'][1] * unit_conversion,
                '+', mew=3, label='BCG', ms=25)

    if 'centroid' in fhat:
       ax.plot(fhat['centroid'][0] * unit_conversion,
               flip_y * fhat['centroid'][1] * unit_conversion,
               'go', mew=3, label='i-band weighted centroid',
                ms=15, fillstyle='none'
                )

    if 'shrink_cent' in fhat:
         ax.plot(fhat['shrink_cent'][0] * unit_conversion,
                 flip_y * fhat['shrink_cent'][1] * unit_conversion,
                 'cx', mew=3,
                 label='shrink_cent', ms=25, fillstyle='none'
                )


    if allPeaks:
        cm = plt.cm.get_cmap('bwr')
        for i in range(len(fhat["peaks_dens"][:])):
            sc = ax.scatter(fhat["peaks_xcoords"][i] * unit_conversion,
                            fhat["peaks_ycoords"][i] * flip_y * unit_conversion,
                            c=fhat["peaks_dens"][i],
                            cmap=cm, vmin=0, vmax=1.0, edgecolor='k',
                            s=35, marker='s')
        if colorbar_ax is not None:
            fig.colorbar(sc, cax=colorbar_ax)

    if showDomPeak:
        ax.plot(fhat["peaks_xcoords"][0] * unit_conversion,
                fhat["peaks_ycoords"][0] * flip_y * unit_conversion,
                's', mew=4., markersize=9, label='dominant KDE peak',
                fillstyle='none', color='gold')

    ax.set_title("Clst {0}: ".format(clstNo) +
                 "No of gal. peaks found = {0}\n".format(
                     len(fhat["peaks_dens"][:])) +
                 "Total peak dens = {0:.3g}".format(
                     np.sum(fhat["peaks_dens"][:])),
                 size=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # rotate tick label
    map(lambda x: x.set_rotation(xlabel_rotate_angle), ax.get_xticklabels())

    if R200C is not None:
        R200_circl = plt.Circle((0, 0), radius=R200C, color='k', lw=2,
                                ls='solid', fill=False, label="R200C")
        plt.plot(0, 0, 'ko', fillstyle='none', label='center of R200C circle',
                 mew=1.5, ms=10)
        # fig = plt.gcf()
        fig.gca().add_artist(R200_circl)

    ax.legend(loc='upper right', frameon=True, numpoints=1, fontsize=14,
              bbox_to_anchor=legend_box_anchor, markerscale=legend_markerscale)


    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ylims = sorted([ylim * flip_y for ylim in ylims])
        ax.set_ylim(ylims)

    low_xlim, up_xlim = ax.get_xlim()
    low_ylim, up_ylim = ax.get_ylim()
    plot_bandwidth_matrix(fhat["bandwidth_matrix_H"][:],
                          up_xlim=0.8 * up_xlim,
                          up_ylim=up_ylim,
                          low_xlim=low_xlim,
                          low_ylim=0.8 * low_ylim,
                          ax=ax,
                          flip_y=flip_y,
                          unit_conversion=unit_conversion
                          )

    if save and clstNo is not None:
        plt.savefig(fileDir + fileName + str(clstNo) + ".png",
                    bbox_inches='tight')

    if showData:
        plt.show()

    return sc


def plot_data_and_peak(df, peaks, R200C=None, save=False, title=None,
                       alpha=0.6, cut=1e3, clstNo=None, units=None):
    """
    :params df: pandas data frame with suitable column names
    :params peaks: np.array, what's spat out from do_KDE_and_get_peaks
    :params R500C: float, the radius to plot a circle to visualize on the plot
    """
    mask = df["SubhaloLenType1"] > 1e3
    plt.axes().set_aspect('equal')
    plt.plot(df.SubhaloPos0[mask], df.SubhaloPos1[mask], '.', alpha=alpha)
    plt.plot(peaks[0], peaks[1], 'rx', mew=2, label="KDE density peak")

    if R200C is not None:
        R200_circl = plt.Circle((0, 0), radius=R200C, color='m', lw=1,
                                ls='solid', fill=False, label="R200C")
        plt.plot(0, 0, 'mo', fillstyle=None, label='center of R200C circle')
        fig = plt.gcf()
        fig.gca().add_artist(R200_circl)

    if title is not None:
        if clstNo is not None:
            title = "clst {0}: ".format(clstNo) + title
        plt.title(title)

    if units is not None:
        plt.xlabel(units)
        plt.ylabel(units)

    plt.legend(loc='best')
    return None


def plot_bandwidth_matrix(mtx, up_xlim, up_ylim, low_xlim, low_ylim, ax,
                          debug=False, unit_conversion=1., flip_y=1):
    """
    :params mtx: numpy array
        represents symmetric positive definite matrix
    """
    eigval, eigvec = np.linalg.eig(mtx)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[order]

    width, height = np.sqrt(eigval) * 2.
    angle = np.arctan2(*eigvec[:, 0][::-1]) / np.pi * 180.
    # want ellipse to be in lower right corner
    mux = up_xlim - 1.5 * width
    muy = (low_ylim + 1.5 * width)

    if debug:
        print( "eigval {0} \n eigvec are {1}".format(eigval, eigvec))
        print( "matrix is {0}".format(mtx))
        print( "width: {0}, height: {1}, angle {2}".format(
            width, height, angle))
        print("centers = ", mux, muy)
    if flip_y == -1:
        angle = -angle
    ell = Ellipse(xy=np.array([mux, muy]),
                  width=width * unit_conversion,
                  height=height * unit_conversion,
                  angle=angle, color="m", edgecolor='none')
    ax.text(mux - 1.7 * width, muy - 1.3 * height, 'kernel size')
    ax = plt.gca()
    ax.add_artist(ell)

    return ell
