"""plot data properties in gal clusters from Illustris-1 simulation
Author: Karen Ng <karenyng@ucdavis.edu>
License: BSD
"""
from __future__ import division
import matplotlib.pyplot as plt
# from matplotlib.mlab import bivariate_normal
import numpy as np
from matplotlib.patches import Ellipse
# import rpy2.robjects as robjects


def plot_color_mag_diag(df, bluer_band, redder_band, band_limit,
                        plot=False, save=False, subhalo_len_lim=1e3,
                        savePath="../plots/", clst=None, verbose=False, *args,
                        **kwargs):
    """
    :parameters:
    df = pandas df of each cluster
    bluer_band = string, df colname
    red_band = string, df colname
    band_limit = float,
        how many band magnitude fainter than BCG do we want to examine
    particleLim = int,
        want to ensure subhalos has at least that many particles

    :returns: None

    :stability: works
    """
    bcg_i = df[redder_band].min()
    mask_i = df[redder_band] < bcg_i + band_limit

    if verbose:
        print "subhalos need at least {0} DM".format(subhalo_len_lim) + \
            " particles to be plotted"
    # examine number of DM particles
    mask_ii = df["SubhaloLenType1"] > subhalo_len_lim

    mask_i = np.logical_and(mask_i, mask_ii)

    g_i = df[bluer_band][mask_i] - df[redder_band][mask_i]

    plt.plot(df[redder_band][mask_i], g_i, "b.", alpha=0.3)
    plt.title("Color-magnitude diagram for".format(clst) +
              " {0} subhalos".format(np.sum(mask_i)))
    plt.ylabel(bluer_band + " - " + redder_band)
    plt.xlabel(redder_band)

    if clst is not None:
        plt.title("Cluster {0}: Color-magnitude diagram for".format(clst) +
                  " {0} subhalos".format(np.sum(mask_i)))

    if save is True:
        assert clst is not None, "arg for clst missing"
        plt.savefig(savePath + "/cm_diagram{0}.png".format(clst),
                    bbox_inches="tight")
    plt.close()
    return


def plot_cf_contour(dens, x, y, lvls=[68, 95], show=False, clabel=False,
                    fill=False):
    """this sort through the density, add them up til they are
    below the required confidence level, then plot

    :param dens: np.array, the density estimate, should integrate to 1
    :param x: np.array, x coord of the density estimate
    :param y: np.array, y coord of the density estimate
    :param lvls: list of floats, denotes percentile

    :returns: None
    """
    d = dens.ravel()
    lvls = np.array(lvls) / 100.
    lvl_vals = np.zeros(len(lvls))
    sums = 0

    d = np.sort(d)  # in ascending order
    d_sum = np.sum(d)

    for j in xrange(d.size):
        sums += d[j]
        for i in range(len(lvls)):
            if sums / d_sum <= 1. - lvls[i]:
                lvl_vals[i] = d[j]

    colors = [(0 / 255., 70 / 255., i / len(lvls)) for i in range(len(lvls))]

    # the plt.contour function is weird, if you don't transpose
    # the density, the plotted density will be rotated by 180 clockwise
    if fill is False:
        CS = plt.contour(x, y, dens.transpose(), lvl_vals, linewidths=(1, 1),
                         colors=colors)
    else:
        CS = plt.pcolor(x, y, dens.transpose(), cmap=plt.cm.Blues)

    if clabel:
        str_lvls = {l: "{0:.1f}".format(s * 100)
                    for l, s in zip(CS.levels, lvls)}
        plt.clabel(CS, CS.levels, fmt=str_lvls, inline=1, fontsize=6.5)

    if show:
        plt.show()

    return


def plot_KDE_peaks(fhat, lvls=range(2, 100, 10), allPeaks=False,
                   plotDataPoints=False, save=False, R200C=None,
                   fileName="KDE_plot_cluster", clstNo=None,
                   clabel=False, showData=False, xlabel="x (kpc / h)",
                   ylabel="y (kpc / h)", showDomPeak=True,
                   fileDir="../plots/", fill=False, showContour=True):
    """make a plot of the fhat along with other important info
    :param fhat:
    """

    plt.axes().set_aspect('equal')

    if showContour:
        plot_cf_contour(fhat["estimate"],
                        fhat["eval_points"][0], fhat["eval_points"][1],
                        lvls=lvls, clabel=clabel, fill=fill)


    if plotDataPoints:
        plt.plot(fhat["data_x"].transpose()[0],
                 fhat["data_x"].transpose()[1], 'r.', alpha=1)

    low_xlim, up_xlim = plt.xlim()
    low_ylim, up_ylim = plt.ylim()
    plot_bandwidth_matrix(fhat["bandwidth_matrix_H"],
                          up_xlim=up_xlim, up_ylim=up_ylim,
                          low_xlim=low_xlim, low_ylim=low_ylim)

    if allPeaks:
        cm = plt.cm.get_cmap('winter')
        for i in range(len(fhat["peaks_dens"])):
            sc = plt.scatter(fhat["peaks_xcoords"][i],
                             fhat["peaks_ycoords"][i],
                             c=fhat["peaks_dens"][i],
                             cmap=cm, vmin=0, vmax=1.0)
        plt.colorbar(sc)

    if showDomPeak:
        plt.plot(fhat["peaks_xcoords"][0],
                 fhat["peaks_ycoords"][0],
                 'ro', mew=1, label='dominant KDE peak',
                 fillstyle='none')

    plt.title("Clst {0}: ".format(clstNo) +
              "No of peaks found = {0}".format(len(fhat["peaks_dens"])))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)

    if R200C is not None:
        R200_circl = plt.Circle((0, 0), radius=R200C, color='m', lw=1,
                                ls='solid', fill=False, label="R200C")
        plt.plot(0, 0, 'm.', fillstyle='none', label='center of R200C circle',
                 mew=2)
        fig = plt.gcf()
        fig.gca().add_artist(R200_circl)

    plt.legend(loc='best')
    if showData:
        plt.show()

    if save and clstNo is not None:
        plt.savefig(fileDir + fileName + str(clstNo) + ".png")
        # , bbox_inches='tight')

    return


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


def plot_bandwidth_matrix(mtx, up_xlim, up_ylim, low_xlim, low_ylim):
    """
    :params mtx: numpy array
        represents symmteric positive definite matrix
    """
    eigval, eigvec = np.linalg.eig(mtx)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[order]

    width, height = np.sqrt(eigval) * 2.
    angle = np.arctan2(*eigvec[:, 0][::-1]) / np.pi * 180.
    # want ellipse to be in lower right corner
    mux = up_xlim - .7 * width
    muy = low_ylim + .7 * width

    print "width: {0}, height: {1}, angle {2}".format(width, height, angle)
    ell = Ellipse(xy=np.array([mux, muy]), width=width, height=height,
                  angle=angle, color="m", edgecolor='none')
    ax = plt.gca()
    ax.add_artist(ell)

    return
