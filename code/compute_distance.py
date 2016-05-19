"""
This contains module for computing distances between fhat_star & fhat (from
DM).
"""
import numpy as np
import sys
sys.path.append("../")
import get_DM_centroids as getDM

def convert_DM_path_to_star_path(DM_clstPath, star_key_no=6):
    """
    :DM_clstPath: string, slash separated paths in HDF5 file
    :returns: string, star_clstPath
    """
    return '/'.join(DM_clstPath.split('/')[:star_key_no])


def compute_euclidean_dist(data, origin=None):
    """
    between numpy array of (nobs, ndim) and one point
    if origin doest not represent a data point, results may be wrong.

    :param data: numpy array, shape = (nobs, ndim)
    :param origin: numpy array (optional), shape = (1, ndim )
    :return: float

    > assert origin.shape[1] == data.shape[1]
    """
    if type(data) is not np.ndarray:
        data = np.array(data)

    if origin is None:
        if len(data.shape) == 1:
            data = data.reshape(1, data.shape[0])
        origin = np.zeros(data.shape[1])
    data = data - origin

    if data.ndim > 1:
        return np.array([np.sqrt(np.dot(data[i], data[i])) for i in
                        range(data.shape[0])])
    else:
        return np.sqrt(np.dot(data, data))


def compute_distance_between_DM_and_gal_peaks(
        fhat_star, fhat, fhat_star_to_DM_coord_conversion=1. / 0.704):
    """
    Parameters
    ===========
    fhat_star: OrderedDict, one of the fhat_stars from `get_KDE`
        coordinates from fhat_star is in kpc / h,
        convert this to kpc by multiplying fhat_star_coordinates
    fhat: OrderedDict, fhat output from `getDM.make_histogram_with_some_resolution`

    Returns
    =======
    (dist, ixes) : a tuple of two arrays
        dist: np.array,
        the distance between the fhat_star peaks and the fhat peaks
        ixes: np.array,
        the index in fhat that corresponds to the closest match
        for fhat_star peaks
    DM_peak_no: int, number of significant DM peaks that were considered when
                finding the nearest neighbor
    gal_peak_no: int, number of significant gal peaks, i.e. peak_dens > 0.5
    """
    from scipy.spatial import KDTree
    from collections import OrderedDict

    good_threshold, gal_peak_no = \
        getDM.apply_peak_num_threshold(fhat_star["peaks_dens"], fhat)

    DM_peak_no = getDM.find_num_of_significant_peaks(fhat["peaks_dens"],
                                                     good_threshold)

    valid_DM_peak_coords = np.array([fhat["peaks_xcoords"][:DM_peak_no],
                                     fhat["peaks_ycoords"][:DM_peak_no]]
                                    ).transpose()
    tree = KDTree(valid_DM_peak_coords)

    star_peak_coords = np.array([fhat_star["peaks_xcoords"][:gal_peak_no],
                                 fhat_star["peaks_ycoords"][:gal_peak_no]]
                                ).transpose()
    # Convert kpc / h from stellar peak coords to kpc
    star_peak_coords *= fhat_star_to_DM_coord_conversion

    (dist, DM_ixes) = tree.query(star_peak_coords, k=1, p=2)
    output = OrderedDict({})
    output["dist"] = dist
    output["DM_ixes"] = DM_ixes
    output["gal_peak_no"] = gal_peak_no
    output["DM_peak_no"] = DM_peak_no

    # We use Euclidean distance for our query, i.e. p=2.
    return (output["dist"],
            output["DM_ixes"]),\
            output["gal_peak_no"],\
            output["DM_peak_no"],\
            good_threshold


# ----- convert output to original dictionary form for visualization -------

def retrieve_cluster_path(h5file):
    """
    :param h5file: hdf5 filestream, for fhat objects
    """
    path_lists = []
    def append_corect_path(path_to_be_examined, path_lists):
        if 'estimate' in path_to_be_examined:
            p = '/'.join(path_to_be_examined.split('/')[:-1])
            path_lists.append(p)

    h5file.visititems(lambda x, y: append_corect_path(x, path_lists))
    return path_lists


def get_gpBy_star_objects(star_df, no_of_gal_keys=6):
    """
    This helps massage the df objects to a similar form as a dictionary form

    """
    gal_gp_keys = tuple(star_df.keys()[-no_of_gal_keys:])
    star_gpBy = star_df.groupby(gal_gp_keys, as_index=False)

    return star_gpBy


def get_gpBy_DM_objects(DM_df,  no_of_DM_keys=8):
    """
    This helps massage the df objects to a similar form as a dictionary form

    The groupby keys should be:
        [u'peaks_xcoords', u'peaks_ycoords', u'peaks_dens', u'clstNo', u'cut',
        u'weights', u'los_axis', u'xi', u'phi']

    """
    return get_gpBy_star_objects(DM_df, no_of_gal_keys=no_of_DM_keys)


def retrieve_metadata_from_fhat_as_path(h5_fhat):
    """
    the metadata retrieved this way can be used for doing `groupby`
    :h5_fhat: hdf5 file stream to the fhat file

    :returns: metadata, a list of strings that represent the metadata in the
    correct order
    """
    paths = []

    # this traverses all the possible paths but
    # our path size is small enough it is very fast
    h5_fhat.visit(paths.append)
    path = paths[-1].split('/')
    metadata = []
    this_p = ''
    for p in path:
        this_p += "/" + p
        metadata.append(h5_fhat[this_p].attrs['info'])

    return metadata


def DM_h5path_to_groupbykey(h5path):
    """
    :h5path: string, hdf5 path to a certain object
    :returns: tuple of strings
    """
    path_list = h5path.split("/")
    path_list[0] = int(path_list[0])
    return tuple(path_list)


def DM_h5path_to_star_groupbykey(DM_clstPath, keys_to_include):
    """
    :DM_clstPath: TODO
    :returns: tuple of strings, star_gpBy_key
    """
    path = convert_DM_path_to_star_path(DM_clstPath, keys_to_include).split("/")
    path[0] = int(path[0])
    return tuple(path)


def combine_DM_df_and_h5_to_dict(gpBy, h5fhat, h5path):
    """
    stateful: no
    """
    gpbykeys = DM_h5path_to_groupbykey(h5path)
    # May want to convert some of the quantities back to float!?
    fhat = {k: v for k, v in gpBy.get_group(gpbykeys).iteritems()}

    fhat["estimate"] = h5fhat[h5path]["estimate"][:]
    eval_points = [h5fhat[h5path]["eval_points0"][:],
                   h5fhat[h5path]["eval_points1"][:]]
    fhat["eval_points"] = eval_points
    return fhat


def combine_star_df_and_h5_to_dict(star_gpBy, star_h5fhat, DM_h5path,
                                   keys_to_include):
    """
    :param star_gpBy: pandas groupby object from the star df
    :param star_h5fhat: hdf5 file stream objects
    :param DM_h5path: string, hdf5 file path to corresponding DM cluster object
    :param keys_to_include: int, how many keys to include in the groupby
    """
    gpbykeys = DM_h5path_to_star_groupbykey(DM_h5path, keys_to_include)
    star_h5path = str(gpbykeys[0]) + '/' + '/'.join(gpbykeys[1:])
    # May want to convert some of the quantities back to float!?
    fhat = {k: v for k, v in star_gpBy.get_group(gpbykeys).iteritems()}

    for k in star_h5fhat[star_h5path].keys():
        fhat[k] = star_h5fhat[star_h5path][k][:]

    return fhat


def get_DM_dict_from_run(DM_df, DM_fhat, DM_h5path, DM_gpBy_key_no=8):
    DM_gpBy = get_gpBy_DM_objects(DM_df, DM_gpBy_key_no)
    return combine_DM_df_and_h5_to_dict(DM_gpBy, DM_fhat, DM_h5path)


def get_star_dict_from_run(star_df, star_fhat, DM_h5path, star_gpBy_key_no):
    star_gpBy = get_gpBy_star_objects(star_df, star_gpBy_key_no)
    return combine_star_df_and_h5_to_dict(
        star_gpBy, star_fhat, DM_h5path, star_gpBy_key_no)


# def compute_distance_for_h5_outputs(
#         DM_df, star_df, no_of_DM_keys=9, no_of_gal_keys=6,
#         fhat_star_to_DM_coord_conversion=1./0.704
#     ):
#     """compute the distance for the peak_df HDF5 file
#
#     :DM_df: pandas dataframe, peak_df object from HDF5 files
#     :star_df: pandas dataframe, peak_df
#     :returns: TODO
#     """
#     # from scipy.spatial import KDTree
#
#     DM_gpBy = get_gpBy_DM_objects(DM_df, no_of_DM_keys=no_of_DM_keys)
#     star_gpBy = \
#         get_gpBy_star_objects(star_df, no_of_gal_keys=no_of_gal_keys)
#
#     return


