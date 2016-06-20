""" Various functions for inferring centroids of galaxy population
"""
from __future__ import division
from collections import OrderedDict
import numpy as np
import pandas as pd
from get_KDE import *
from compute_distance import compute_euclidean_dist
from multiprocessing import Pool
# import calculate_astrophy_quantities as cal_astrophy
# import logging

# logging.basicConfig(filename="Debug_get_gal_centroids.log",
#                     level=logging.DEBUG)


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


def cut_dim_galaxies(df, limiting_mag_band="apparent_i_band",
                          limiting_mag=24.4):
    """ make observationally realistic cuts to galaxies
    :df: pandas dataframe containing subhalos of one cluster
    :limiting_mag_band: str, "*_band" that is part of the pandas dataframe
    :limiting_mag: float, max. magnitude in number for galaxies to be
    observable
    :returns: numpy array of booleans

    :note: limiting magnitude of i < -17 is calculated base on
    apparent magnitude < 24 and a luminosity distance of z = 0.3
    """

    return np.array(df[limiting_mag_band] < limiting_mag)


def prep_data_with_cuts_and_wts(df, cuts, cut_methods, cut_cols, wts,
                                verbose=True):
    """
    :param df: pandas dataframe containing all subhalos for each cluster
    :param cut_methods: dict, key is the name of the cut, value is a function
        for judging whether a subhalo is included in the KDE
    :param cut_cols: dict, key is the name of the cut that it is for, value is
        a string that corresponds to the col in the df for judging whether a
        subhalo pass the cut or not
    :param wts: dict, key is the name of the col in the df, value is the
        function that we will apply to the col in the df

    Returns
    -------
    df_list : list of pandas dataframe
    """
    dfs_with_cuts = OrderedDict({})
    richness = OrderedDict({})
    for cut_method_name, cut_kwargs in cuts.iteritems():
        if cut_kwargs is not None:  # skip if cut_kwargs is None
            # apply the cut_methods FUNCTION to dataframe
            mask = cut_methods[cut_method_name](df, **cut_kwargs)
            richness[cut_method_name] = np.sum(mask)
            if verbose:
                print "# of subhalos after the " + \
                    "{1} cut = {0}".format(np.sum(mask), cut_method_name)
            # col = cut_cols[cut_method_name]
            thisdf = df[mask].copy()  # avoid funny behavior by hard copying
        else:
            if verbose:
                print "there is no cut. # of subhalos = {}".format(
                    df.shape[0])
            thisdf = df

        for weight, wt_func in wts.iteritems():
            # example weight could be i-band weighted
            if wt_func:  # if wt_func is not None
                thisdf[weight + "_wt"] = wt_func(thisdf[weight])
            else:
                thisdf[weight + "_wt"] = np.ones(thisdf.shape[0])

        # have new df for each types of cuts
        dfs_with_cuts[cut_method_name] = thisdf.copy()

    return dfs_with_cuts, richness


def mag_to_lum(mag):
    return np.exp(-mag + 23.)


def convert_Mpc_over_h_to_Mpc(coords):
    return coords * 106.5 / 75.


# --------- functions for computing gal-DM offsets-------------------------
def compute_KDE_offsets(peak_xcoords, peak_ycoords):
    """
    :param fhat: dictionary
        with `peak_xcoords` and `peak_ycoords` obtained from
        `do_KDE_and_get_peaks`

    :return: list of [offset, offsetR200]
        offset: offset in unit of c kpc/h
    :to do:
        * needs major revamp to restructure the outputs, i.e. fhat
        * may want to check the dimension of input data
    :note:
        can think of making this function even more general
        by having the peak inference function passed in
    """

    # we have sorted the density so that the highest density peak is the first
    # entry in the peak_*coords array

    # need the array version of the computation
    if (type(peak_xcoords) == np.ndarray and
        type(peak_ycoords == np.ndarray)) or \
        (type(peak_xcoords) == pd.core.series.Series and
            type(peak_ycoords) == pd.core.series.Series):
        peaks = np.array([peak_xcoords, peak_ycoords]).transpose()
    elif type(peak_xcoords) == float and type(peak_ycoords) == float:
        # and the non-array version of the computation
        peaks = np.array([peak_xcoords, peak_ycoords])

    return compute_euclidean_dist(peaks)


def compute_KDE_R200Coffsets(offset, f, clstNo):
    """
    :param f: hdf5 file stream object
        connecting to the Illustris SUBFIND catalog
    :param clstNo: integer

    :return: offsetR200
        offset: offset in terms of the R200C of the cluster
    :note:
        can think of making this function even more general
        by having the peak inference function passed in
    """

    # each cluster only have one R200C
    R200C = f["Group"]["Group_R_Crit200"][clstNo]

    return offset / R200C


def compute_shrinking_aperture_offset(
        df, f, clstNo, cut_method, cut_kwargs,
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

        offset: offset is in unit of kpc/h
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


def find_gal_peak_ixes_in_DM_fhat(star_peaks_xcoord, star_peaks_ycoord, fhat):
    """
    Uncomment the following to check the type before running!
    assert type(star_peaks_xcoord) == np.float64, \
        "`star_peaks_xcoord` needs to be of type `np.float64`."

    assert type(star_peaks_ycoord) == np.float64, \
        "`star_peaks_ycoord` needs to be of type `np.float64`."

    assert type(fhat["eval_points"][0]) == np.ndarray, \
        "`fhat['eval_points'][0]` needs to be of type `np.ndarray`."

    assert type(fhat["eval_points"][1]) == np.ndarray, \
        "`fhat['eval_points'][1]` needs to be of type `np.ndarray`."

    The resulting indices will introduce an uncertainty of ~2 kpc
    due to the bin size of fhat["eval_points"]
    """

    x_ix = 0
    while (fhat["eval_points"][0][x_ix] < star_peaks_xcoord):
        x_ix += 1

    y_ix = 0
    while (fhat["eval_points"][1][y_ix] < star_peaks_ycoord):
        y_ix += 1

    return x_ix, y_ix


# ---------- Utilities for converting dictionaries to h5 objects -------
def convert_dict_peaks_to_df(
    fhat, metadata, save=False, output_path="../data/",
    peak_h5="fhat_peak.h5", peak_info_keys=None,
    ):
    """
    :param fhat_list: list of python dictionary obtained from
        `convert_rfhat_to_dict`
    :param weights_used: string, metadata about the data set

    :return: df
    """
    if peak_info_keys is None:
        peak_info_keys = ["peaks_xcoords",
                          "peaks_ycoords",
                          # "peaks_rowIx",
                          # "peaks_colIx",
                          "peaks_dens"]

    peak_df = pd.DataFrame()
    for key in peak_info_keys:
        peak_df[key] = fhat[key]

    # Starts storing meta data
    for key, val in metadata.iteritems():
        peak_df[key] = val

    return peak_df


def construct_h5_file_for_saving_fhat(metadata, dens_h5,
                                      output_path="../../data/"):
    """
    constructs the skeleton of the hdf5 file using an OrderedDict to capture
    its structure.

    metadata is assumed to have the following keys:
        * "clstNo"
        * "cut"
        * "weights"
        * "los_axis"
        * "xi"
        * "phi"
    :metadata: OrderedDict, keys correspond to the projections
    :dens_h5: h5 file stream
    """
    import h5py
    import collections

    if type(metadata) != collections.OrderedDict:
        raise TypeError("metadata needs to be an OrderedDict.")

    h5_fstream = h5py.File(output_path + dens_h5,
                           mode="a", compression="gzip",
                           compression_opts=9)

    # Would implement this recursively if the data structure were more regular
    for clstNo in metadata["clstNo"]:
        lvl1 = h5_fstream.create_group(str(clstNo))

        # we put the "key" that corresponds to a dictionary of cuts
        # as the meta data
        for cuts in metadata["cut"].keys():
            lvl2 = lvl1.create_group(cuts)

            # we put the "key" that corresponds to what band to weight
            # the data by as the meta data
            for weights in metadata["weights"].keys():
                lvl3 = lvl2.create_group(weights)

                for los_axis in metadata["los_axis"]:
                    lvl4 = lvl3.create_group(str(los_axis))

                    # should probably create 'xi_phi' as one group
                    # or make sure there is no empty group
                    for projection in metadata["projection"]:
                        try:
                            lvl5 = lvl4.create_group(str(projection))
                        except ValueError:
                            print(
                                "ValueError raised due to creating existing groups")

                        # for phi in np.unique(metadata["phi"]):
                        #     try:
                        #         lvl5.create_group(str(phi))
                        #     except ValueError:
                        #         print(
                        #             "ValueError raised due to creating existing groups")

    # add metadata to the largest entry of each group!
    lvl1.attrs['info'] = metadata.keys()[0]  # clstNo
    lvl2.attrs['info'] = metadata.keys()[1]  # cuts
    lvl3.attrs['info'] = metadata.keys()[2]  # weights
    lvl4.attrs['info'] = metadata.keys()[3]  # los_axis
    lvl5.attrs['info'] = metadata.keys()[4]  # projection
    # lvl5[np.unique(metadata["phi"])[-1]].attrs['info'] = metadata.keys()[5]

    return h5_fstream


def metakeys():
    """keys with which our h5 files are organized
    this is somewhat a documentation, not really for use
    """
    return ["clstNo", "cut", "weights", "los_axis", "xi", "phi"]


def metadata_from_h5_key_path(h5_key_path):
    from collections import OrderedDict
    keys = ["clstNo", "cut", "weights", "los_axis", "xi", "phi"]
    val = h5_key_path.split("/")
    return OrderedDict({k: val[i] for i, k in enumerate(keys)})


def retrieve_fhat_from_gp(gp_keys, gp_vals, h5_fhat_fstream):
    """
    :param gp_val: pandas dataframes containing peak info
    :param gp_keys: groupby keys from a dictionary
    :param h5_fhat_fstream: hdf5 file stream
        to the data output with projections, cuts and weights

    :returns:
        a dictionary of fhat objects
    :example
        >>>fhat_dict = {gp_keys:
                        getg.retrieve_fhat_from_gp(gp_keys, gp_vals, h5_fstream)
                        for gp_keys, gp_vals in groups.iteritems()}
    """
    path = [str(v) + "/" for v in gp_keys]
    path = ''.join(path)

    fhat_keys = ["estimate", "bandwidth_matrix_H", "eval_points"]
    fhat = {key: h5_fhat_fstream[path + key][:] for key in fhat_keys}

    peak_keys = ["peaks_dens", "peaks_xcoords", "peaks_ycoords"]
    for k in peak_keys:
        fhat[k] = gp_vals[k]

    return fhat


def h5path_from_clst_metadata(clst_metadata):
    path = [str(v) + "/" for v in clst_metadata.values()]
    return ''.join(path)


def convert_dict_dens_to_h5(fhat, clst_metadata, h5_fstream,
                            fixed_size_data_keys=[
                                "eval_points", "estimate", "bandwidth_matrix_H",
                                "shrink_cent", "centroid", 'BCG'
                            ]):
    """
    :param fhat: dictionary, key is cluster property, value is the value
    :param clst_metadata: OrderedDict,
        key is key for the metadata, value is the
        attribute values
    :param h5_fstream: hdf5 file stream
    :param

    :note: stateful, stuff gets written to the hdf5 file
    :returns: None
    """

    path = h5path_from_clst_metadata(clst_metadata)

    for k in fixed_size_data_keys:
        h5_fstream[path + k] = fhat[k]

    return


def find_3D_peaks():
    # find
    # needs to check 27 - 7 points from the cube
    return


def get_subhalo_mass(df):
    return np.array(df)

# -----------other centroid methods -----------------------------
def shrinking_apert(data, center_coord=None, r0=None, debug=False, w=None):
    """
    :param center_coord: list of floats or array of floats
    :param data: numpy array
        with shape[1] == center_coord.shape[0] == len(w)
        shape[0] = number of observations
    :param r0: float, aperture to consider in the data
    :param debug: bool, output debugging messages or not
    :param w: float in numpy array like form, or in pandas dataframe,
       denotes the weights for each galaxy data point
    :returns: numpy array,
        with same shape as center_coord
    :note: I want to write this procedure so that it would work
        in both 2D and 3D
    """

    data, normalization = normalize_data(data)

    if w is None:
        w = np.ones(len(data))
    elif len(w) != len(data):
        raise ValueError(
            "length mismatch between data `data` and weights `w`")

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


def get_BCG_location(df, band=None, spat_key1="SubhaloPos0",
                     spat_key2="SubhaloPos1"):
    ix = np.argmin(df[band])
    spat_cols = [spat_key1, spat_key2]
    return np.array(df[spat_cols].iloc[ix])


def get_BCG_ixes(df, DM_cut=1e3, star_cut=1e2,
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


def get_gal_coords(df, DM_cut=1e3, star_cut=1e2, coordkey0="SubhaloPos1",
                   coordkey1="SubhaloPos1"):
    # from scipy.spatial import KDTree

    df = df[cut_reliable_galaxies(df, DM_cut, star_cut)]
    coords = np.array(df[[coordkey0, coordkey1]])

    return coords


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


# --------- compute confidence region for each method ------------------

def angles_give_same_projections(phi_arr, xi_arr):
    """
    :param phi_arr: np array of floats, in RADIANS, azimuthal angle, 0 to 2 * np.pi
    :param xi_arr: np array of floats, in RADIANS, elevation angle, 0 to np.pi

    :returns: an array of bools
    """
    from itertools import combinations

    # The number of comparisons needed is nC2
    combo = np.array([c for c in combinations(range(len(xi_arr)), 2)])
    return np.array([same_projection(phi_arr[pair[0]], xi_arr[pair[0]],
                                     phi_arr[pair[1]], xi_arr[pair[1]])
                    for pair in combo]), np.array(combo)


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
    :param xi: float, elevation angle in radian
    :param phi: float, azimuthal angle in radian
    :param los_axis: integer, line-of-sight (los) axis, 0 = x, 1 = y, 2 = z

    :return: array like object as coords, same dimension
    """

    xi = float(xi)
    phi = float(phi)
    los_axis = int(los_axis)

    if not radian:
        xi = xi / 180. * np.pi
        phi = phi / 180. * np.pi

    from numpy import cos, sin
    # rotate our view point, origin is at (0, 0, 0)
    # rotate by azimuthal angle phi first then elevation angle xi
    mtx = np.array([[cos(phi) * cos(xi), -cos(xi)*sin(phi), sin(xi)],
                    [sin(phi), cos(phi), 0],
                    [-cos(phi) * sin(xi), sin(phi) * sin(xi), cos(xi)]
                    ])
    # mtx = np.array([[cos(phi)*cos(xi), -sin(phi), cos(phi)*sin(xi)],
    #                 [sin(phi)*cos(xi), cos(phi), sin(phi)*sin(xi)],
    #                 [-sin(xi), 0, cos(xi)]])

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


def spherical_coord_to_cartesian(phi, xi):
    """
    :ref:
        https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates
    """
    return np.array([np.sin(xi) * np.cos(phi),
                    np.sin(xi) * np.sin(phi),
                    np.cos(xi)
                     ])


def angles_given_HEALpix_nsides(nside):
    """
    :param nside: integer, this integer must be a power of 2, int_val = 2 ** n
    :returns: tuple of two arrays, each array corresponds to angles in radians
        phi_arr
        xi_arr

    (At least) half the projected outputs are identical because we cannot tell
    apart "front" and "back" for a projection.
    """
    from healpy import pix2ang
    from healpy.pixelfunc import nside2npix

    npix = nside2npix(nside)
    angle_idxes = range(npix)
    xi_arr, phi_arr = pix2ang(nside, angle_idxes, nest=False)

    # Find duplicate projection by brute force
    sameP, combo = angles_give_same_projections(phi_arr, xi_arr)
    first_half, second_half = combo[sameP].transpose()

    return phi_arr[first_half], xi_arr[first_half],


def get_clst_gpBy_from_DM_metadata(metadata_df, gpBy_keys=None):
    """
    :metadata_df: pandas dataframe, retrieved from `fhat_star_*_peak_df.h5` file
    :gpBy_keys: list of strings, column names to groupBy

    :returns: groupby object, gpBy_df
        example usage:
        >>> gps = gpBy_df.groups.keys()
        >>> gpBy_df.get_group(gps[0])
    """
    if gpBy_keys is None:
        gpBy_keys = list(metadata_df.keys()[-6:])

    print ("gpBy_keys = ", gpBy_keys)
    print ("len(gpBy_keys) = ", len(gpBy_keys))
    return metadata_df.groupby(gpBy_keys, as_index=False), gpBy_keys


def same_projection(phi1, xi1, phi2, xi2):
    """
    :phi1: float, azimuthal angle in radians
    :xi1: float, elevation angle in radians
    :phi2: float, azimuthal angle in radians
    :xi2: float, elevation angle in radians

    Determine if two sets of angles correspond to the same projection.

    We test for azimuthal symmetry because we project w.r.t. z-axis.

    :tests: passed

    """
    if phi1 == phi2 and xi1 == xi2:
        return True
    elif np.abs(xi1 + xi2 - np.pi) < np.finfo(np.float32).eps and \
        np.abs(np.abs(phi1 - phi2) - np.pi) < np.finfo(np.float32).eps:
        return True
    else:
        return False
        # Brute force
        # pt1 = project_coords(np.array([1, 2, 3])/np.sqrt(14.), xi1, phi1)
        # pt2 = project_coords(np.array([1, 2, 3])/np.sqrt(14.), xi2, phi2)
        # pt0 = np.zeros(3)
        # dist1 = compute_euclidean_dist(pt1, pt0)
        # dist2 = compute_euclidean_dist(pt2, pt0)
        # same_projection = np.abs(dist1 - dist2) < np.finfo(np.float32).eps


    # return same_projection

# --------- depreciated R conversion functions ------------------------------

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



