""" various functions for inferring centroids of galaxy population
"""
from __future__ import division
from collections import OrderedDict
import numpy as np
import pandas as pd
from get_KDE import *


# --------- functions for preparing cuts, projections, weighting -----------

def cut_reliable_galaxies(df, DM_cut=1e3, star_cut=1e2):
    """ consider all cluster galaxies with minimal cuts
    :params df: pandas dataframe contains one cluster
    :params DM_cut: integer, how many DM particles needed for us to consider
        subhalos to be reliable
    :params star_cut: integer, how many stellar particles needed for us to
        consider subhalos to be reliable
    :returns: numpy array of booleans

    :usage:
        >>> cut_reliable_galaxies(df, **cut_kwargs)

    :notes:
    http://illustris-project.org/w/index.php/Data_Details#Snapshot_Contents
    """
    return np.array(np.logical_and(df["SubhaloLenType1"] > DM_cut,
                                   df["SubhaloLenType4"] > star_cut))


def prep_data_with_cuts_and_wts(df, cuts, cut_methods, cut_cols, wts,
                                verbose=False):
    """
    :param df: pandas dataframe containing all subhalos for each cluster
    :param cut_methods: function

    Returns
    -------
    df_list : list of pandas dataframe
    """
    dfs_with_cuts = OrderedDict({})
    for cut_method_name, cut_kwargs in cuts.iteritems():
        mask = cut_methods[cut_method_name](df, **cut_kwargs)
        if verbose:
            print "# of subhalos after the " + \
                  "{1} cut = {0}".format(np.sum(mask), cut_method_name)
        # col = cut_cols[cut_method_name]
        thisdf = df[mask].copy()

        for weight, wt_func in wts.iteritems():
            if weight != "no":
                thisdf[weight + "_wt"] = wt_func(thisdf[weight])
            else:
                thisdf[weight + "_wt"] = np.ones(thisdf.shape[0])


        # have new df for each types of cuts
        dfs_with_cuts[cut_method_name] = thisdf.copy()

    return dfs_with_cuts


def mag_to_lum(mag):
    return np.exp(-mag + 23.)


def project_cluster_df(df, nside_pow=None, los_axis=2, verbose=True):
    """
    :param df: pandas dataframe containing all subhalos for each cluster
    :param nside_pow: (optional) integer, nside = 2 ** nside_pow
        This should be a number between [0, 30]
        this decides how many pixels of the projections will have.
        The smaller nside_pow is, the less number of pixels / projections
        the function computes.
    """

#     if nside_pow is not None:
#         nside = 2 ** nside_pow
#
#     coords = map(lambda xi, phi: project_coords(df, xi, phi,
#                                                 los_axis=los_axis,
#                                                 radian=True),
#                  xi_array, phi_array)
#     return coords, xi_array, phi_array



# --------- functions for computing gal-DM offsets-------------------------
def compute_KDE_peak_offsets(fhat, f, clstNo):
    """
    :param fhat: dictionary
        with `peak_xcoords` and `peak_ycoords` obtained from
        `do_KDE_and_get_peaks`
    :param f: hdf5 file stream object
        connecting to the Illustris SUBFIND catalog
    :param clstNo: integer

    :return: list of [offset, offsetR200]
        offset: offset in unit of c kpc/h
        offset: offset in terms of the R200C of the cluster
    :to do:
        * needs major revamp to restructure the outputs, i.e. fhat
        * may want to check the dimension of input data
    :note:
        can think of making this function even more general
        by having the peak inference function passed in
    """

    # fhat = do_KDE_and_get_peaks(data, w=w)

    # each cluster only have one R200C
    R200C = f["Group"]["Group_R_Crit200"][clstNo]

    # we have sorted the density so that the highest density peak is the first
    peaks = np.array(fhat["peaks_xcoords"][0], fhat["peaks_ycoords"][0])
    fhat["offset"] = np.sqrt(np.dot(peaks, peaks))
    fhat["offsetR200"] = fhat["offset"] / R200C

    return


def compute_shrinking_aperture_offset(df, f, clstNo, cut_method, cut_kwargs,
                                      w=None, verbose=True,
                                      col=["SubhaloPos0", "SubhaloPos1"],
                                      projection=None, output='both'):
    """
    :param df: pandas dataframe for each cluster
    :param cut_method: function
    :param w: floats, weight
    :param col: list of strings, the strings should be df keys
    :param projection: 2-tuple of floats, (theta, phi), not yet implemented

    :return:
        if output is 'both'
        list of [1st_spatial_peak_loc, 2nd_spatial_peak_loc, offset]

        offset: offset is in unit of c kpc/h



    :to do:
        needs major revamp to restructure the outputs, i.e. fhat
    :note:
        can think of making this function even more general
        by having the peak inference function passed in
    """
    if output not in ['both', 'peaks_only', 'offsets_only']:
        raise NotImplementedError("Implemented output types are " +
                                  "`both`, `peaks_only`, `offsets_only`")
    mask = cut_method(df, **cut_kwargs)
    if verbose:
        print "# of subhalos after the cut = {0}".format(np.sum(mask))

    data = np.array(df[col][mask])

    if w is not None:  # make sure the weight dimensions match data
        w = np.array(w)  # convert pandas weights to np.array
        w = w[mask]

    shrink_cent = shrinking_apert(data, w=w)
    offset = compute_euclidean_dist(shrink_cent)

    if shrink_cent.ndim == 2 and output == 'both':
        return np.array([shrink_cent[0], shrink_cent[1], offset])
    elif output == 'peaks_only':
        return shrink_cent
    elif output == 'offsets_only':
        return offset
    else:
        raise NotImplementedError(
            "Haven't generalized to compute offset for ndim > 2")


# ---------- Utilities for converting dictionaries to h5 objects -------
def convert_dict_peaks_to_df(fhat, metadata,
                             save=False, output_path="../data/",
                             peak_h5="fhat_peak.h5"):
    """
    :param fhat_list: list of python dictionary obtained from
        `convert_rfhat_to_dict`
    :param weights_used: string, metadata about the data set

    :return: df
    """
    # try:
    #     old_df = pd.read_hdf(output_path + peak_h5, wt + "df")
    #     append = True
    # except IOError:
    #     append = False

    peak_df_list = []
    peak_info_keys = ["peaks_xcoords", "peaks_ycoords", "peaks_rowIx",
                      "peaks_colIx", "peaks_dens"]

    peak_df = pd.DataFrame()
    for key in peak_info_keys:
        peak_df[key] = fhat[key]
    # starts storing meta data

    for key, val in metadata.iteritems():
        peak_df[key] = val

    # peak_df_list.append(peak_df.copy())

    # peak_df = pd.concat(peak_df_list, axis=0)

    # if append:
    #     peak_df = pd.concat(old_df, peak_df, axis=0)
    #     peak_df = peak_df.drop_duplicates()

    # if save:
    #     peak_df.to_hdf(output_path + peak_h5, wt + "_df", complevel=9,
    #                    complib="zlib")

    return peak_df


def convert_dict_dens_to_h5(fhat_list, meta_data,
                            save=False, output_path="../data/",
                            dens_h5="fhat_dens.h5"):
    """
    INCOMPLETE : needs one more subgroup for clstNo
    """

    import h5py
    fixed_size_data_keys = ["eval_points", "estimate", "bandwidth_matrix_H"]
    f = h5py.File(output_path + dens_h5, mode="a", compression="gzip",
                  compression_opts=9)

    # first create the group that tells us which weighting (wt) scheme it uses
    try:
        subgroup = f.create_group(wt)
    except ValueError:
        print ("ValueError was raised due to create existing group")
        subgroup = f[wt]

    # create projection
    try:
        subgroup1 = subgroup.create_group(str(phi))
    except ValueError:
        print ("ValueError was raised due to create existing group")
        subgroup1 = f[wt + "/" + str(phi)]
    try:
        subgroup2 = subgroup1.create_group(str(xi))
    except ValueError:
        print ("ValueError was raised due to create existing group")
        subgroup2 = f[wt + "/" + str(phi) + "/" + str(xi)]

    # then create the names of each element as a subgroup
    try:
        map(subgroup2.create_group, fixed_size_data_keys)
    except ValueError:
        print ("ValueError was raised due to create existing subgroups")

    for i, fhat in enumerate(fhat_list):
        for fkeys in fixed_size_data_keys:
            # the final key is the clstNo
            key = wt + "/" + str(phi) + "/" + str(xi) + "/"  + \
                fkeys + "/" + str(i)
            f[key] = fhat[fkeys]

    f.close()
    return


def find_3D_peaks():
    # find
    # needs to check 27 - 7 points from the cube
    return


# -----------other centroid methods ------------------------------------
def shrinking_apert(data, center_coord=None, r0=None, debug=False, w=None):
    """
    :param center_coord: list of floats or array of floats
    :param data: numpy array
        with shape[1] == center_coord.shape[0]
        shape[0] = number of observations
    :param r0: float, aperture to consider in the data

    :returns: numpy array,
        with same shape as center_coord
    :note: I want to write this procedure so that it would work in both 2D and
    3D
    """

    data, normalization = normalize_data(data)

    if w is None:
        w = np.ones(len(data))
    elif len(w) != len(data):
        raise ValueError("length mismatch between data `data` and weights `w`")

    if center_coord is not None:
        c1 = np.array(center_coord)
        # We don't want to worry about different scales of the data.
        c1 = c1 / normalization
    else:
        # Start with mean of the data.
        # Should not start with the origin because that is cheating.
        c1 = compute_weighted_centroids(data, w=w)

    dist = compute_euclidean_dist(data - c1)

    assert c1.shape[0] == data.shape[1], "dimension mismatch between " + \
        "center and data"

    if r0 is None:
        r0 = np.percentile(dist, 90)
        if debug:
            print "no r0 was given, setting r0 to {0}".format(
                r0 * compute_euclidean_dist(normalization))
    else:
        assert r0 > 0, "initial aperture has to be greater than 0"

    mdist = np.mean(dist)
    c0 = c1 + 10 * np.mean(dist)
    mask = dist < r0
    it = 0

    conseq_c1_diff = []
    conseq_c1 = []
    while(np.abs(compute_euclidean_dist(c1 - c0) - mdist) / mdist > 2e-2
          and np.sum(mask) > 10):
        # compute new convergence criteria
        mdist = compute_euclidean_dist(c1 - c0)
        if debug:
            it += 1
            print "iter {0} : c1 = {1}, data no = {2}".format(it, c1,
                                                              np.sum(mask))
            print "mdist = {0}".format(mdist)
        c0 = c1
        # compute new centroid
        c1 = compute_weighted_centroids(data[mask], w=w[mask])
        dist = compute_euclidean_dist(data - c1)  # compute new dist
        r0 *= 0.95  # shrink the aperture
        mask = dist < r0
        if mdist == 0:
            break

        if debug:
            print "(new mdist - old mdist) / old mdist = {0}\n".format(
                np.abs(compute_euclidean_dist(c1 - c0) - mdist) / mdist)
            conseq_c1_diff.append(np.abs(
                compute_euclidean_dist(c1 - c0) - mdist) / mdist)
            conseq_c1.append(c1)

    if debug:
        return c1 * normalization, conseq_c1 * normalization, conseq_c1_diff
    else:
        return c1 * normalization


def normalize_data(data):
    """
    :param data: numpy array,
        first dim should be the observation number
    """
    if type(data) is not np.ndarray:
        data = np.array(data)

    if data.ndim > 1:
        normalization = np.array([data[:, i].max() - data[:, i].min() for i in
                                 range(data.shape[1])])
    else:
        normalization = data.max() - data.min()

    assert np.any(normalization != 0), \
        "Range of data in at least one dimension is zero, \
        please check your data"

    return data / normalization, normalization


def compute_euclidean_dist(data):
    """
    :param data: numpy array
    :return: numpy array
    """
    if type(data) is not np.ndarray:
        data = np.array(data)

    if data.ndim > 1:
        return np.array([np.sqrt(np.dot(data[i], data[i])) for i in
                        range(data.shape[0])])
    else:
        return np.sqrt(np.dot(data, data))


def compute_weighted_centroids(x, w=None):
    """
    :param x: numpy array, the data,
    :param w: numpy array, the weights
    """
    if w is None:
        return np.mean(x, axis=0)
    elif w.ndim == 1:
        w = w.reshape(w.shape[0], 1)

    if len(x) != len(w):
        raise ValueError("length of data and weights have to be the same")
    return np.sum(x * w, axis=0) / np.sum(w)


def get_BCG_ix(df, DM_cut=1e3, star_cut=1e2,
               bands=None, verbose=False, brightest_only=True):
    """
    :param df: pandas dataframe, contains all subhalos of each cluster
    :param DM_cut: (optional) integer,
        min. no. of DM particles that we require for a subhalo to be qualified
        as a galaxy
    :param gal_cut: (optional) integer,
        min. no. of star particles that we require for a subhalo to be
        qualified as a galaxy
    :param bands: (optional) list of strings, each string should be a key
        in the df. If not provided, default redder bands are used.
    :param brightest_only: (optional) boolean, if only the galaxy that is
    brighter in more number of bands is returned

    :returns: the row index of the BCG in the dataframe
        if the BCG is not the brightest in all bands,
        the galaxy that are the brightest in most number of bands is returned
    """
    from scipy.stats import mode
    # sort by U, B, V, K, g, r, i, z bands
    # magnitude : brighter = smaller magnitude
    df = df[cut_reliable_galaxies(df, DM_cut, star_cut)]

    if bands is None:
        bands = ['r_band', 'i_band', 'z_band', 'K_band']

    ixes = np.argmin(np.array(df[bands]), axis=0)

    if len(np.unique(ixes)) == 1 and verbose:
        print("BCG is consistently the brightest in all bands")
        # return ixes[0]
    elif verbose:
        print("BCG is not consistently the brightest in all bands\n" +
              "no of bright galaxies = {0}".format(np.unique(ixes)))
    if brightest_only:
        return mode(ixes)[0][0]
    else:
        # if we want more than 1 brightest galaxies - return
        # all galaxies that are the brightest in any of the 4 bands
        return np.unique(ixes)


def get_BCG_offset(df, spat_key1="SubhaloPos0", spat_key2="SubhaloPos1",
                   cut_kwargs={"DM_cut": 1e3, "star_cut": 1e2},
                   bands=None, verbose=False):
    """
    :param df: pandas dataframe with dataframe containing
        subhalos of each clusters, i.e. output from calling extract_clst from
        using list comprehension
    :param phi: (NOT IMPLEMENTED) float, azimuthal angle
    :param xi: (NOT IMPLEMENTED) float, elevation angle
    :param cut_kwargs: (optional) dictionary of cuts parameters,
        that should be used to ensure subhalos correspond to galaxies
    :param bands: (optional) list of strings that correspond to the name of the
        bands for use to determine which galaxy is a BCG
    :param verbose: (optional) boolean, any message should be printed

    :returns: scalar value / vector of scalars of the offset
    """
    ix = get_BCG_ix(df, bands=bands, **cut_kwargs)
    BCG_offset = \
        np.array(df[[spat_key1, spat_key2]].iloc[ix])
    return compute_euclidean_dist(BCG_offset)


def galaxies_closest_to_peak(df, list_of_coord_keys, peak_coords,
                             k_nearest_neighbor=None):
    from scipy.spatial import KDTree

    if k_nearest_neighbor is None:
        if int(df.shape[0] * 0.05) > 5:
            k_nearest_neighbor = int(df.shape[0] * 0.05)
        else:
            k_nearest_neighbor = 3

    tree = KDTree(np.array(df[list_of_coord_keys]))
    if type(peak_coords) != np.ndarray:
        peak_coords = np.array(peak_coords)

    # we use Euclidean distance for our query, i.e. p=2
    return tree.query(peak_coords, k=k_nearest_neighbor, p=2)

# ---------- weights --------------------------------------------------

# --------- compute confidence region for each method ------------------

def get_shrinking_apert_conf_reg(data_realizations):
    """
    :param data_realizations: list of data sets
        each generated from same distribution

    :return fhat: dictionary of properties
        of shrinking aperture peaks
        from the data realizations and the peaks
    """
    # find shrinking aperture peak of each realization
    shrink_peaks2 = np.array([shrinking_apert(bi_data)
                             for bi_data in data_realizations])

    # use KDE to find the distribution of the shrinking aperture peaks
    shrink_peak_dens2 = do_KDE(shrink_peaks2)
    shrink_peak_dens2 = convert_rfhat_to_dict(shrink_peak_dens2)
    find_peaks_from_py_diff(shrink_peak_dens2)

    return shrink_peak_dens2


def get_centroid_conf_reg(data_realizations):
    """
    :param data_realizations: list of data_realizations

    :return fhat: dictionary of properties
        of KDE peaks from the data realizations and the peaks
    """
    cent_fhat2 = [compute_weighted_centroids(g_data)
                  for g_data in data_realizations]
    cent_peak_dens2 = do_KDE(cent_fhat2)
    cent_peak_dens2 = convert_rfhat_to_dict(cent_peak_dens2)
    find_peaks_from_py_diff(cent_peak_dens2)

    return cent_peak_dens2


# --------- compute projection matrix ----------------------------------
def los_axis_to_vector(los_axis):
    """return a vector for dot product projction"""
    return np.arange(3) != los_axis


def project_coords(coords, xi, phi, los_axis=2, radian=True):
    """
    :param coords: array like / df
    :param xi: float, elevation angle in degree
    :param phi: float, azimuthal angle in degree
    :param los_axis: integer, line-of-sight (los) axis, 0 = x, 1 = y, 2 = z

    :return: same type of array like object as coords, same dimension
    """
    if not radian:
        xi = xi / 180. * np.pi
        phi = phi / 180. * np.pi

    from numpy import cos, sin
    # rotate our view point, origin is at (0, 0, 0)
    mtx = np.array([[cos(phi)*cos(xi), -sin(phi), cos(phi)*sin(xi)],
                    [sin(phi)*cos(xi), cos(phi), sin(phi)*sin(xi)],
                    [-sin(xi), 0, cos(xi)]])

    if type(coords) != np.ndarray:
        coords = np.array(coords)

    proj_plane = los_axis_to_vector(los_axis)
    if coords.ndim == 1:
        # we do the rotation of the view point before projecting
        # to a lower dimension
        return proj_plane * np.dot(mtx, coords)
    elif coords.ndim > 1:
        data = np.array(map(lambda d: proj_plane * np.dot(mtx, d), coords))
        return data


def same_projection(phi1, xi1, phi2, xi2):
    """
    determine if two sets of angles correspond to the same projection
    """
    raise NotImplementedError
    return


def angles_given_HEALpix_nsides(nside):
    """
    :param nside: integer, must be powers of 2
    :returns: tuple of two arrays, each array corresponds to xi and phi values
        values are in radians

    angle_idxes convert indexes such as
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    to
    [0, 1, 4, 5, 8, 9]
    so we only want to sample half the pixels
    """
    from healpy import pix2ang
    from healpy.pixelfunc import nside2npix

    npix = nside2npix(nside)
    # we only want half the pixels on the sphere due to symmetry
    angle_idxes = np.array([range(2 * (i - 1), 2 * (i - 1) + 2)
                            for i in np.arange(1, int(npix / 2), 2)]
                           ).ravel()  # flatten array
    xi, phi = pix2ang(nside, angle_idxes)

    return xi, phi


# --------- meta data handling -----------------------------------------

# def convert_R_peak_ix_to_py_peaks(fhat, ix_key="peak_coords_ix",
#                                   pt_key="eval_points"):
#     """
#     :param fhat: python dictionary
#     :returns coords: numpy array
#         shape=(n_peak, 2)
#
#     :stability: this should be tested!
#     """
#     py_peak_ix = fhat[ix_key] - 1  # python is zeroth indexed
#     return np.array([[fhat[pt_key][0, ix[0]], fhat[pt_key][1, ix[1]]]
#                      for ix in py_peak_ix])
#
# def do_KDE(data, bw_selector="Hscv", w=None, verbose=False):
#     """
#     :param data: np.array, with shape (dataNo, 2)
#     :param bw_selector: str, either 'Hscv' or 'Hpi'
#     :param w: np.array of floats that denote the weight for each data point
#     :param verbose: bool
#
#     :return: fhat, ks R object spat out by KDE()
#     """
#     assert data.shape[1] == 2, \
# "data array is of the wrong shape, want array with shape (# of obs, 2)"
#     assert bw_selector == 'Hscv' or bw_selector == 'Hpi', \
#         "bandwidth selector {0} not available".format(bw_selector)
#
#     data = py_2D_arr_to_R_matrix(data)
#     doKDE = robjects.r["do_KDE"]
#
#     if w is not None:
# needs to be tested
#         fhat = doKDE(data, robjects.r[bw_selector],
#                      robjects.FloatVector(w))
#     else:
#         fhat = doKDE(data, robjects.r[bw_selector])
#
#     return fhat
#
# def get_peaks(fhat, no_of_peaks=1):
#     """
#     :param fhat: robject spat out from ks.KDE
#     :param no_of_peaks: integer
#
#     :returns: list of peaks,
#         each row correspond to one peak [[x1, y1], [x2, y2], ...]
#     """
#     findPeaks = robjects.r["find_peaks_from_2nd_deriv"]
#     findDomPeaks = robjects.r["find_dominant_peaks"]
#
#     peaks_ix = findPeaks(fhat)  # fhat[2] = fhat$estimate
#     dom_peaks = np.array(findDomPeaks(fhat, peaks_ix, no_of_peaks))
#
#     fhat = convert_rfhat_to_dict(fhat)
#     # subtracting 1 from the peak_coords since python is zeroth index,
#     # R has 1 as the first index
#     fhat["peaks_py_ix"] = np.array(peaks_ix) - 1
#
#     # need to double check how the coordinates are put into vectors
#     # something might have been transposed ...
#     fhat["domPeaks"] = dom_peaks
#
#     return fhat


# def find_peaks_from_2nd_deriv(fhat, verbose=False):
#     """not tested but works without errors
#     fhat = robject returned by ks.KDE
#     """
#     func = robjects.r["find_peaks_from_2nd_deriv"]
#
#     return func(fhat, verbose)
