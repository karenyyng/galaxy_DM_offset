"""for inferring DM centroids / peaks """
from __future__ import (print_function, division)
import numpy as np
import matplotlib.pyplot as plt
import get_KDE
from scipy.spatial import KDTree


def make_histogram_with_2kpc_resolution(data, coord_key="coords",
                                        spatial_axis=range(2),
                                        close_plot=True):
    """this function makes histogram and returns appropriate format
    this should be used

    :param data: data dictionary
        obtained from `extract_catalog.get_DM_particles()`

    key-value pairs
    ---------------
    :key coords: numpy array
        shape is (n_observation, n_spatial_dimension)
        this array should have a min. coord value of 0.0 or else 2d histogram
        will fail
    :key min_coord: numpy array
        shape is (1, n_spatial_dimension)

    :param coord_key: str
        key to the data dictionary for getting the value of the coord array
    :param spatial_axis: list of two integers
        integer should represent the index of the spatial axis for making
        histogram

    :note: Illustris 1 DM particle resolution is 1.42 kpc

    """
    # compute bin numbers for each spatial dimension with 2 kpc resolution
    bins = np.array(map(lambda d: int((int(np.max(d)) / 2.)),
                    data[coord_key].transpose()))

    fhat = {}
    fhat["estimate"], edges1, edges2, image = \
        plt.hist2d(data[coord_key][:, spatial_axis[0]],
                   data[coord_key][:, spatial_axis[1]],
                   bins=bins[spatial_axis], cmap=plt.cm.BrBG)

    edges = [edges1, edges2]
    # compute center of histogram bins
    # then add the min. coordinate that we subtracted before to avoid negative
    # values, now the coordinates will be in the original frame
    fhat["eval_points"] = np.array([0.5 * (edges[i][1:] + edges[i][:-1]) +
                                    data["min_coords"][spatial_axis[i]]
                                    for i in range(2)])

    get_KDE.find_peaks_from_py_diff(fhat)
    get_KDE.get_density_weights(fhat)

    if close_plot:
        plt.close()
    else:
        plt.show()

    return fhat


def match_DM_peaks_with_gal_peaks(fhat, fhat_stars, threshold=0.3,
                                  convert_kpc_over_h_to_kpc=True,
                                  verbose=True):
    """
    Parameters
    ----------
    fhat : dictionary
        contains all the peak information of the DM density
        this dict. is the output from `make_histogram_with_2kpc_resolution`
    fhat_stars : dictionary
        contains all the peak information of the galaxies
        (weighted / unweighted).
        This dict. is the output of `get_gal_centroids.do_KDE_and_get_peak()`
    threshold : float
        the DM peak density threshold for peaks to be considered in the
        matching process.
    convert_kpc_over_h_to_kpc : bool
        whether to convert the gal fhat coordinates from kpc / h to kpc
    verbose : bool
        print info or not

    Return
    ------
    dist : numpy array of floats
        distance in kpc of the matched object from the gal peak
        len(dist) = len(fhat_stars["peaks_dens"])
    match : numpy array of integers
        index of the matched object in the masked DM peaks dictionary
        corresponding to the closest match to the gal peak.
        e.g. [3, 1, 2, 4]
        would mean the 3rd DM peak matches to the 1st gal peak
        1st DM peak matches to the 2nd gal peak etc.
        len(match) = len(fhat_stars["peaks_dens"])
    """
    # only consider peaks over a certain density threshold
    peaks_mask = fhat["peaks_dens"] > threshold
    DMpeakCoords = np.array([fhat["peaks_xcoords"][peaks_mask],
                            fhat["peaks_ycoords"][peaks_mask]]).transpose()
    tree = KDTree(DMpeakCoords)

    galpeakCoords = np.array([fhat_stars["peaks_xcoords"],
                              fhat_stars["peaks_ycoords"]]).transpose()

    if verbose:
        print("Converting subhalo distance units from kpc / h to kpc")

    if convert_kpc_over_h_to_kpc:
        galpeakCoords *= 106.5 / 75.

    dist, match = tree.query(galpeakCoords)

    return dist, match


# ------------ unstable but may be used if all else fails -------------------

def get_dens_and_grid(x, y, bw='normal_reference',
                      gridsize=100, cut=4,
                      clip=[-np.inf, np.inf], n_jobs=10):
    """wrapper around statsmodel and seaborn function for inferring 2D density
    :note: unstable:
    """
    from seaborn.distributions import _kde_support
    import statsmodels.nonparametric.kernel_density as KDE
    KDEMultivariate = KDE.KDEMultivariate

    kde = KDEMultivariate(np.array([x, y]), var_type='cc', bw=bw)
    kde.n_jobs = n_jobs

    x_support = _kde_support(x, kde.bw[0], gridsize, cut, clip)
    y_support = _kde_support(y, kde.bw[1], gridsize, cut, clip)
    xx, yy = np.meshgrid(x_support, y_support)

    z = kde.pdf([xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, z
