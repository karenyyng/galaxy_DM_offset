"""
This contains module for computing distances between fhat_star & fhat
(from DM).
"""
# import h5py
import pandas as pd
import numpy as np
import sys
sys.path.append("../")
# import extract_catalog as ec
import compute_distance as compDist
import get_DM_centroids as getDM


def construct_uber_result_df(star_fhats, DM_fhats, main_h5):
    """uses different functions for constructing the uber result dataframe

    :param star_fhats: hdf5 filestream, output from test_star_fhat_*.py
    :param DM_fhats: hdf5 filestream, output from test_DM_fhat_*.py
    :param main_h5:
        Illustris data filestream that contains metadata info about each
        cluster
    :returns: uber_df,
        a Pandas dataframe that contains the main result of the paper
    """
    clstNo = [int(no) for no in star_fhats.keys()]

    # Do not combine dataframe with projections in fhat objects
    # until the very very end
    uber_df = pd.DataFrame([])
    uber_df['clstNo'] = sorted(clstNo)
    uber_df["M200C"] = main_h5['Group/Group_M_Crit200'][
        uber_df['clstNo']]

    paths = retrieve_cluster_path(star_fhats,
                                  property_key="peaks_dens")
    const_path = '/'.join(paths[0].split('/')[1:3])
    uber_df["richness"] = [
        star_fhats[str(no) + '/' + const_path + "/" + "richness"
                   ].value
        for no in sorted(clstNo)
    ]

    return uber_df


def compute_distance_from_most_bound_particle(
    star_fhats, DM_fhats, compute_2D_distance=False,
    bin_widths=['/0.0/', '/25.0/'], star_paths=None,
    summary_stat_keys=['centroid', 'BCG', 'shrink_cent'],
    save=False, filename=None, primary_peak=True
    ):
    """
    :star_fhats: h5 file stream

 :DM_fhats: h5 file stream
    :compute_2D_distance: bool
    :returns: TODOm
    """
    df_list = []

    if star_paths is None:
        star_paths = compDist.retrieve_cluster_path(star_fhats)

    for star_path in star_paths:
        star_fhat = star_fhats[star_path]

        temp_dict = {key + "_" + direction: star_fhat[key][i]
                     for key in summary_stat_keys
                     for i, direction in enumerate(['x', 'y'])}

        # temporarily put results in matched_stat first
        # this is useful if we want to compute secondary peak locations
        # matched_stat = compDist.compute_distance_between_DM_and_gal_peaks(
        #         star_fhat, DM_fhat, compute_2D_distance=False
        #     )

        star_props = star_path.split('/')
        temp_dict['projection'] = star_props[-1]
        temp_dict['clstNo'] = int(star_props[0])

        if primary_peak:
            for direction in ['x', 'y']:
                temp_dict["lum_KDE_{}".format(direction)] = \
                    star_fhat['peaks_{}coords'.format(direction)][0]
        else:
            raise NotImplementedError(
                "Duplicate rows for non-primary peaks needed.")

        df_list.append(pd.DataFrame(temp_dict, index=[temp_dict['clstNo']]))

    mbp_df = pd.concat(df_list)

    return mbp_df


def convert_result_fhat_to_proj_uber_df(
    star_fhats, DM_fhats, compute_2D_distance=True,
    bin_widths=['/0.0/', '/25.0/'], star_paths=None,
    summary_stat_keys=['centroid', 'BCG', 'shrink_cent'],
    save=False, filename=None
    ):
    """
    :star_fhats: h5 file stream
    :DM_fhats: h5 file stream
    :compute_2D_distance: bool
    :returns: TODO
    """
    df_list = []

    if star_paths is None:
        star_paths = compDist.retrieve_cluster_path(star_fhats)

    for star_path in star_paths:
        clstNo = [int(star_path.split('/')[0])]
        star_fhat = star_fhats[star_path]
        for bin_width in bin_widths:
            DM_fhat = DM_fhats[star_path + bin_width]

            # temporarily put results in matched_stat first
            matched_stat = compDist.compute_distance_between_DM_and_gal_peaks(
                    star_fhat, DM_fhat, compute_2D_distance=True
                )

            # compute all other offsets before putting them somewhere
            # for computing my uber_df
            dist_dict = compDist.compute_distance_for_other_peaks(
                matched_stat, star_fhat, summary_stat_keys=summary_stat_keys,
                compute_2D_distance=True
            )
            # gal_peak_no is the number of significant luminosity peaks
            # that we match
            peak_no = matched_stat['gal_peak_no']
            if peak_no > 1:
                df = pd.DataFrame(dist_dict,
                                  index=[clstNo[0] for i in range(peak_no)])
            else:
                df = pd.DataFrame(dist_dict, index=clstNo)

            # can think of modifying the data type of matched_stat...
            df['peak_id'] = range(peak_no)
            df['KDE' ] = matched_stat['dist']
            df['Delta_x_KDE'] = matched_stat['Delta_x_KDE']
            df['Delta_y_KDE'] = matched_stat['Delta_y_KDE']
            df['matched_DM_peak_x'] = \
                matched_stat['DM_matched_peak_coords']['peaks_x_coords']
            df['matched_DM_peak_y'] = \
                matched_stat['DM_matched_peak_coords']['peaks_y_coords']

            df['total_peaks_dens'] = np.sum(star_fhat['peaks_dens'])
            # bin_widths contain strings in format of '/bin_width/'
            df['bin_width'] = float(bin_width[1:-1])
            df['gal_peak_no'] = matched_stat['gal_peak_no']
            df['projection'] = star_path.split('/')[-1]
            df_list.append(df)

    # this is rbind
    uber_df_proj = pd.concat(df_list)

    if save and filename is None:
        raise ValueError('`filename` cannot be None')
    elif save and filename is not None:
        uber_df_proj.to_hdf(filename,'df')

    return uber_df_proj


def assign_sign_for_dist():
    if np.random.randint(10) % 2 == 0:
        return 1.
    else:
        return -1.


def compute_distance_for_other_peaks(
        matched_stat, star_fhat, summary_stat_keys,
        unit_conversion=1./0.704, convert_kpc_over_h_to_kpc=True,
        compute_2D_distance=False
    ):
    """use output from `compute_distance_between_DM_and_gal_peaks` to
    compute the distances for other summary stat

    :param matched_stat: output dictionary from
        `compute_distance_between_DM_and_gal_peaks`, quantities are in units of
        kpc
    :param star_fhat: hdf5 file stream object, contains the gal density info
        output from `test_star_fhat_*.py`, quantities are in units of kpc / h
    :param summary_stat_keys: string,
        the key in star_fhat for the relevant stat info
    :param returns: dictionary
        float, distance in kpc for that particular summary stat indicated by
        the key

    :note: if there are several DM peaks, this function returns the shortest
    distance to the DM peak.
    """
    from scipy.spatial import KDTree

    if type(summary_stat_keys) is str:
        summary_stat_keys = list(summary_stat_keys)
    elif type(summary_stat_keys) is not list:
        raise TypeError("summary_stat_keys needs to be a list of str")

    if not convert_kpc_over_h_to_kpc:
        unit_conversion = 1.

    dist_dict = {}
    for sum_stat_key in summary_stat_keys:
        star_sum_xy_coords = star_fhat[sum_stat_key][:] * unit_conversion
        valid_DM_peak_coords = np.array([
            matched_stat["DM_matched_peak_coords"]['peaks_x_coords'],
            matched_stat["DM_matched_peak_coords"]['peaks_y_coords']]
        ).transpose()

        tree = KDTree(valid_DM_peak_coords)
        (dist_dict[sum_stat_key], DM_ixes) = \
            tree.query(star_sum_xy_coords, k=1, p=2)

        if compute_2D_distance:
            dist_dict["Delta_x_" + sum_stat_key], \
            dist_dict["Delta_y_" + sum_stat_key] = \
                np.array(star_sum_xy_coords - valid_DM_peak_coords[DM_ixes]
                        ).transpose()

    return dist_dict


def convert_DM_path_to_star_path(DM_clstPath, star_key_no=-1):
    """
    :DM_clstPath: string, slash separated paths in HDF5 file
    :returns: string, star_clstPath
    """
    return '/'.join(DM_clstPath.split('/')[:star_key_no])


def compute_dist_between_matched_DM_peak_and_no_dens_peak(
        star_fhat_no_dens, uber_df,
        fixed_star_path='/mag/None/1/'):
    """Docstring for compute_dist_between_matched_DM_peak_and_no_dens_peak.

    :star_fhat_no_dens: h5stream object
    :uber_df: pandas dataframe like object
    :star_paths: list of strings, paths to no density peak estimates
    :returns: TODO

    """
    from scipy.spatial import KDTree
    if 'clstNo' not in uber_df:
        uber_df['clstNo'] = np.array(uber_df.index)

    uber_df.index = range(uber_df.shape[0])

    star_paths = uber_df.apply(
        lambda x: str(x.clstNo) + fixed_star_path + str(x.projection),
        axis=1
    )

    # slow to preallocate .... it is ok
    outputs = {"no_dens_dist": [],
               "Delta_no_peak_x": [],
               "Delta_no_peak_y": [],
               "no_dens_xcoord": [],
               "no_dens_ycoord": [],
               }

    # have to build a KDTree for each entry
    for i, path in enumerate(star_paths):
        no_peak_coords = np.array([star_fhat_no_dens[path]['peaks_xcoords'],
                                   star_fhat_no_dens[path]['peaks_ycoords']]
                                  ).transpose()

        tree = KDTree(no_peak_coords)
        (dist, ix) = tree.query(
            tuple(uber_df.ix[i, ['matched_DM_peak_x', 'matched_DM_peak_y']]),
            k=1, p=2)
        outputs["no_dens_dist"].append(dist)
        outputs["Delta_no_peak_x"].append(
            star_fhat_no_dens[path]['peaks_xcoords'][ix] -
            uber_df.ix[i, 'matched_DM_peak_x']
        )
        outputs["Delta_no_peak_y"].append(
            star_fhat_no_dens[path]['peaks_ycoords'][ix] -
            uber_df.ix[i, 'matched_DM_peak_y']
        )
        outputs["no_dens_x"].append(
            star_fhat_no_dens[path]['peaks_xcoords'][ix])
        outputs["no_dens_y"].append(
            star_fhat_no_dens[path]['peaks_ycoords'][ix])

    return pd.DataFrame(outputs)


def compute_euclidean_dist(data, origin=None):
    """
    between numpy array of (nobs, ndim) and one point
    if `origin` param doest not represent a data point, results may be wrong.

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
        fhat_star, fhat, fhat_star_to_DM_coord_conversion=1. / 0.704,
        verbose=False, compute_2D_distance=False):
    """
    Parameters
    ===========
    fhat_star: OrderedDict, one of the fhat_stars from `get_KDE`
        coordinates from fhat_star is in kpc / h,
        convert this to kpc by multiplying fhat_star_coordinates
    fhat: OrderedDict,
        fhat output from `getDM.make_histogram_with_some_resolution`

    Returns
    =======
    An ordered dictionary, the key-value pairs are as follows:
    dist: np.array,
        the distance between the fhat_star peaks and the fhat peaks
    ixes: np.array,
        the index in fhat that corresponds to the closest match
        for fhat_star peaks
    DM_peak_no: int, number of significant DM peaks that were considered when
                finding the nearest neighbor
    gal_peak_no: int, number of significant gal peaks, i.e. peak_dens > 0.2
    star_sign_peak_coords: dictionary of two numpy arrays, the coordinates are
        in terms of kpc. These coordinates denote where the significant KDE
        luminosity peaks are used for matching the DM peaks.
    DM_matched_peak_coords: dictionary of two numpy arrays, the coordinates are
        in terms of kpc. These coordinates denote which DM peaks were matched
        to significant luminosity peaks. There is a one-to-one correspondance
        between the `star_sign_peak_coords` and `DM_matched_peak_coords` for
        the computation of the distances returned below.
    Delta_x_KDE: float or numpy array of floats, depending on how many peaks
        were used for the matching. This denotes a difference between
        'star_sign_peak_coords' and `DM_matched_peak_coords` along the x-axis.
    Delta_y_KDE: float or numpy array of floats, depending on how many peaks
        were used for the matching. This denotes a difference between
        'star_sign_peak_coords' and `DM_matched_peak_coords` along the y-axis.
    """
    from scipy.spatial import KDTree
    from collections import OrderedDict

    gal_peak_no = \
        getDM.find_num_of_significant_peaks(fhat_star["peaks_dens"][:],
                                            threshold=0.5
                                            )

    DM_peak_no = \
        getDM.apply_peak_num_threshold(fhat_star["peaks_dens"][:], fhat,
                                       verbose=verbose
                                       )

    valid_DM_peak_coords = np.array([fhat["peaks_xcoords"][:DM_peak_no],
                                     fhat["peaks_ycoords"][:DM_peak_no]]
                                    ).transpose()

    if verbose:
        print ("gal_peak_no = ", gal_peak_no)
        print ("DM_peak_no = {0}".format(DM_peak_no))
        print ("valid_DM_peak_coords = ", valid_DM_peak_coords)

    star_peak_coords = np.array([fhat_star["peaks_xcoords"][:gal_peak_no],
                                 fhat_star["peaks_ycoords"][:gal_peak_no]]
                                ).transpose()
    # Convert kpc / h from stellar peak coords to kpc
    star_peak_coords *= fhat_star_to_DM_coord_conversion

    # We use Euclidean distance for our query, i.e. p=2.
    tree = KDTree(valid_DM_peak_coords)
    (dist, DM_ixes) = tree.query(star_peak_coords, k=1, p=2)

    output = OrderedDict({})

    # a test case can check if sqrt(Delta_x_KDE^2 + Delta_y_KDE^2) == dist
    if compute_2D_distance:
        output["Delta_x_KDE"], output["Delta_y_KDE"] = \
            np.array(star_peak_coords - valid_DM_peak_coords[DM_ixes]
                     ).transpose()

    output["dist"] = dist

    output["DM_ixes"] = DM_ixes
    output["DM_matched_peak_coords"] = {
        "peaks_x_coords": fhat["peaks_xcoords"][:][DM_ixes],
        "peaks_y_coords": fhat["peaks_ycoords"][:][DM_ixes],
    }
    output["star_sign_peak_coords"] = {
        "peaks_x_coords":
        fhat_star["peaks_xcoords"][:][:gal_peak_no] *
        fhat_star_to_DM_coord_conversion,
        "peaks_y_coords":
        fhat_star["peaks_ycoords"][:][:gal_peak_no] *
        fhat_star_to_DM_coord_conversion,
    }
    output["gal_peak_no"] = gal_peak_no
    output["DM_peak_no"] = DM_peak_no

    return output

# - convert output to original dictionary form for visualization -------
def append_correct_path(path_to_be_examined, path_lists,
                        property_key="peaks_dens"):
    if property_key in path_to_be_examined:
        p = '/'.join(path_to_be_examined.split('/')[:-1])
        path_lists.append(p)


def retrieve_cluster_path(h5file, property_key='peaks_dens'):
    """
    :param h5file: hdf5 filestream, for fhat objects
    :param property_key: string, the key to look for in the path
    """
    path_lists = []

    h5file.visit(lambda x:
                 append_correct_path(x, path_lists, property_key))
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

    last_clstNo = sorted([int(clstNo) for clstNo in h5_fhat.keys()])[-1]

    correct_paths = []
    h5_fhat[str(last_clstNo)].visititems(
        lambda x, y: append_correct_path(x, correct_paths)
    )

    path = [str(last_clstNo)] + correct_paths[-1].split('/')
    metadata = []

    this_p = ''
    for p in path:
        this_p += "/" + p
        metadata.append(h5_fhat[this_p].attrs['info'])

    return metadata


### Compute ranking of the distance



# --- deprecated ----------------------------

# def DM_h5path_to_groupbykey(h5path):
#     """
#     :h5path: string, hdf5 path to a certain object
#     :returns: tuple of strings
#     """
#     path_list = h5path.split("/")
#     path_list[0] = int(path_list[0])
#     return tuple(path_list)
#
#
# def DM_h5path_to_star_groupbykey(DM_clstPath, keys_to_include):
#     """
#     :DM_clstPath: TODO
#     :returns: tuple of strings, star_gpBy_key
#     """
#     path = convert_DM_path_to_star_path(DM_clstPath, keys_to_include).split("/")
#     path[0] = int(path[0])
#     return tuple(path)
#
#
# def combine_DM_df_and_h5_to_dict(gpBy, h5fhat, h5path):
#     """
#     stateful: no
#     """
#     gpbykeys = DM_h5path_to_groupbykey(h5path)
#     # May want to convert some of the quantities back to float!?
#     fhat = {k: v for k, v in gpBy.get_group(gpbykeys).iteritems()}
#
#     fhat["estimate"] = h5fhat[h5path]["estimate"][:]
#     eval_points = [h5fhat[h5path]["eval_points0"][:],
#                    h5fhat[h5path]["eval_points1"][:]]
#     fhat["eval_points"] = eval_points
#     return fhat
#
#
# def combine_star_df_and_h5_to_dict(star_gpBy, star_h5fhat, DM_h5path,
#                                    keys_to_include):
#     """
#     :param star_gpBy: pandas groupby object from the star df
#     :param star_h5fhat: hdf5 file stream objects
#     :param DM_h5path: string, hdf5 file path to corresponding DM cluster object
#     :param keys_to_include: int, how many keys to include in the groupby
#     """
#     gpbykeys = DM_h5path_to_star_groupbykey(DM_h5path, keys_to_include)
#     star_h5path = str(gpbykeys[0]) + '/' + '/'.join(gpbykeys[1:])
#     # May want to convert some of the quantities back to float!?
#     fhat = {k: v for k, v in star_gpBy.get_group(gpbykeys).iteritems()}
#
#     for k in star_h5fhat[star_h5path].keys():
#         fhat[k] = star_h5fhat[star_h5path][k][:]
#
#     return fhat
#
## def get_DM_dict_from_run(DM_df, DM_fhat, DM_h5path, DM_gpBy_key_no=8):
#     DM_gpBy = get_gpBy_DM_objects(DM_df, DM_gpBy_key_no)
#     return combine_DM_df_and_h5_to_dict(DM_gpBy, DM_fhat, DM_h5path)
#
#
# def get_star_dict_from_run(star_df, star_fhat, DM_h5path, star_gpBy_key_no):
#     star_gpBy = get_gpBy_star_objects(star_df, star_gpBy_key_no)
#     return combine_star_df_and_h5_to_dict(
#        star_gpBy, star_fhat, DM_h5path, star_gpBy_key_no)


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


