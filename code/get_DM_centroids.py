"""for inferring DM centroids / peaks """
from __future__ import (print_function, division)
import numpy as np
import matplotlib.pyplot as plt
import get_KDE
from scipy.spatial import KDTree
import compute_distance as compDist


def make_histogram_with_some_resolution(data, resolution=2.0,
                                        coord_key="coords",
                                        spatial_axis=range(2),
                                        close_plot=True, find_peak=True):
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
    :states: no

    """
    # compute bin numbers for each spatial dimension with 2 kpc resolution
    bins = np.array(map(lambda d: int((int(np.max(d)) / resolution)),
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

    if find_peak:
        get_KDE.find_peaks_from_py_diff(fhat)
        get_KDE.get_density_weights(fhat)

    if close_plot:
        plt.close()
    else:
        plt.show()

    return fhat


def match_DM_peaks_with_gal_peaks(fhat, fhat_stars, threshold=0.3,
                                  convert_kpc_over_h_to_kpc=True,
                                  verbose=False, k_neighbors=1,
                                  distance_upper_bound=100,
                                  p_norm=2):
    """
    Parameters
    ----------
    fhat : dictionary
        contains all the peak information of the DM density
        this dict. is the output from `make_histogram_with_some_resolution`
    fhat_stars : dictionary
        contains all the peak information of the galaxies
        (weighted / unweighted).
        This dict. is the output of `get_gal_centroids.do_KDE_and_get_peak()`
    threshold : float
        the DM peak density threshold for peaks to be considered in the
        matching process.
    convert_kpc_over_h_to_kpc : bool
        whether to convert the gal fhat coordinates from kpc / h to kpc
    k_neighbors : int
        number of neighbors per query entry to return
    distance_upper_bound : float
        in kpc, what is the upper bound to return, inf is returned if distance
        of nearest neighbor is outside the bound
    p_norm : integer
        what norm to use. 1-Manhattan distance, 2-Euclidean norm
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

    dist, match = tree.query(galpeakCoords, p=p_norm,
                             distance_upper_bound=distance_upper_bound,
                             k=k_neighbors)

    return dist, match


def retrieve_DM_metadata_from_gal_h5file(gal_fhat_h5file):
    """This retrieves the gal metadata from the appropriate h5 file
    * matches the DM metadata from the gal_metadata so we can compute the
    correct offsets. There are no cuts / weights for the DM particles.

    :gal_fhat_h5file: hdf5 file stream for the star fhat data
    :returns: DM_metadata, this is a ordered dictionary
    """
    from collections import OrderedDict

    metadata_keys = \
        compDist.retrieve_metadata_from_fhat_as_path(gal_fhat_h5file)
    metadata_dict = OrderedDict({})
    metadata_vals = compDist.retrieve_cluster_path(gal_fhat_h5file)
    metadata_vals = np.array([p.split('/') for p in metadata_vals]).transpose()
    metadata_vals = \
        [np.unique(p) for p in metadata_vals[:-1]] + list(metadata_vals[-1:])

    for i, k in enumerate(metadata_keys):
        if k == 'clstNo':
            metadata_dict[k] = [int(v) for v in metadata_vals[i]]
        else:
            metadata_dict[k] = metadata_vals[i]

    metadata_dict['projection'] = [eval(p) for p in
                                   np.unique(metadata_dict['projection'])]

    return metadata_dict


def retrieve_DM_metadata_from_gal_metadata(dataPath, gal_metadata_h5_file,
                                           h5key="peak_df", keys=None):
    """This retrieves the gal metadata from the appropriate h5 file
    * matches the DM metadata from the gal_metadata so we can compute the
    correct offsets. There are no cuts / weights for the DM particles.

    :gal_metadata_h5_file:
    :returns: DM_metadata, this is a ordered dictionary
    :returns: metadata_df,
    """
    import pandas as pd
    from collections import OrderedDict

    metadata_df = pd.read_hdf(dataPath + gal_metadata_h5_file, h5key)

    if keys is None:
        keys = ["clstNo", "cut", "weights", "los_axis",  ("xi", "phi")]

    def retrieve_metadata(metadata_df, group_by_cols):
        return (metadata_df
                .groupby(group_by_cols, as_index=False)
                .groups
                .keys()
                )

    DM_metadata = OrderedDict({})

    for k in keys:
        if type(k) is not tuple:
            DM_metadata[k] = retrieve_metadata(metadata_df, k)
        else:
            temp_data = retrieve_metadata(metadata_df, k)
            temp_data = np.array([list(d) for d in temp_data]).transpose()
            for i, td in enumerate(temp_data):
                DM_metadata[k[i]] = td

    return DM_metadata, metadata_df


def construct_h5_file_for_saving_fhat(metadata, dens_h5,
                                      output_path="../../data/"):
    """
    :metadata: OrderedDict
    :dens_h5: file name for the hdf5 file for storing fhat
    :output_path: directory for storing the h5 file dens_h5

    :returns: hdf5 filestream
    """
    import collections
    if type(metadata) != collections.OrderedDict:
        # we don't need an OrderedDict in this current code structure
        # but should make it more general... so we don't have
        # to specify the keys explicitly below
        raise TypeError("metadata needs to be an OrderedDict.")

    import h5py
    h5_fstream = h5py.File(output_path + dens_h5,
                           mode="a", compression="gzip",
                           compression_opts=9)

    # Would implement this recursively if the data structure were more regular
    # also need to do error handling.
    for clstNo in sorted(np.unique(metadata["clstNo"])):
        lvl1 = h5_fstream.create_group(str(clstNo))

        for cuts in metadata["cut"]:
            lvl2 = lvl1.create_group(cuts)

            for weights in metadata["weights"]:
                lvl3 = lvl2.create_group(weights)

                for los_axis in metadata["los_axis"]:
                    lvl4 = lvl3.create_group(str(los_axis))

                    # more groups are created than needed
                    for projection in metadata["projection"]:
                        try:
                            lvl5 = lvl4.create_group(str(projection))
                        except ValueError:
                            print(
                                "ValueError raised due to creating existing groups")

                        # for phi in np.unique(metadata["phi"]):
                        #     try:
                        #         lvl6 = lvl5.create_group(str(phi))
                        #     except ValueError:
                        #         print(
                        #             "ValueError raised due to creating existing groups")

                        #     # for sig_fraction in metadata["sig_fraction"]:
                        #     #     lvl7 = lvl6.create_group(str(sig_fraction))

                        for kernel_width in np.unique(metadata["kernel_width"]):
                            try:
                                lvl5.create_group(str(kernel_width))
                            except ValueError:
                                print(
                                    "ValueError raised due to creating existing groups")



    # write out the metadata to the smallest value entry of each group:
    lvl1.attrs['info'] = "clstNo"
    lvl2.attrs['info'] = "cut"
    lvl3.attrs['info'] = "weights"
    lvl4.attrs['info'] = "los_axis"
    lvl5.attrs['info'] = "projection"
    # lvl7.attrs['info'] = "sig_fraction"
    lvl5[str(metadata['kernel_width'][-1])].attrs['info'] = "kernel_width"

    return h5_fstream


def convert_dict_dens_to_h5(fhat, clst_metadata, h5_fstream, verbose=False,
                            fixed_size_data_keys=[
                                "eval_points", "estimate", "peaks_xcoords",
                                "peaks_ycoords", "peaks_dens"
                            ]):
    import get_gal_centroids as getgal

    path = getgal.h5path_from_clst_metadata(clst_metadata)
    if verbose:
        print (path)
        print (clst_metadata)

    for k in fixed_size_data_keys:
        if k != "eval_points":
            thispath = path + k
            if verbose:
                print (thispath)
            h5_fstream[thispath] = fhat[k]
        else:
            for i in range(len(fhat[k])):
                thispath = path + k + str(i)
                if verbose:
                    print (thispath)
                h5_fstream[thispath] = fhat[k][i]

    return


def find_num_of_significant_peaks(peak_dens_list, threshold=0.2):
    no_sig_peaks = np.sum(peak_dens_list[1:] > threshold)
    return no_sig_peaks + 1


def apply_peak_num_threshold(gal_peak_dens_list, fhat,
                             multiple_of_candidate_peaks=3,
                             sig_fraction=0.2, verbose=False):
    """
    Set the number of candidate DM peak to be a multiple of the significant
    galaxy peaks.

    Parameters
    -----------
    gal_peak_dens_list : list of floats of relative the KDE peak dens to the
                         densest peak
    fhat : output from `make_histogram_with_some_resolution`
    multiple_of_candidate_peaks: int (optional), default = 2
        how many DM candidate peaks to consider,
            as a multiple of the number of galaxy peaks
    sig_fraction : float, (optional) default = 0.2,
       0 < sig_fraction < 1

    Returns
    -------
    dens_threshold: float, a density threshold for picking peaks
    sig_DM_peaks: integer,
        the number of significant DM peaks to be considered

    """
    if sig_fraction < 0. or sig_fraction > 1.:
        raise ValueError("0 < `sig_fractions` < 1 is required!")

    sig_gal_peaks = find_num_of_significant_peaks(gal_peak_dens_list,
                                                  sig_fraction)

    if sig_gal_peaks >= 3:
        accepted_peak_no = multiple_of_candidate_peaks * sig_gal_peaks
    else:
        accepted_peak_no = multiple_of_candidate_peaks * (sig_gal_peaks + 1)

    # mask = fhat["peaks_dens"] > sig_fraction
    # peaks_dens_above_threshold = np.array(fhat["peaks_dens"][:])[mask]
    # if type(peaks_dens_above_threshold) is np.float64:
    #     no_of_good_peaks = 1
    # elif type(peaks_dens_above_threshold) is np.ndarray:
    #     no_of_good_peaks = len(peaks_dens_above_threshold)
    # if (no_of_good_peaks > accepted_peak_no):
    #     return no_of_good_peaks
    if (len(fhat["peaks_dens"]) < accepted_peak_no):
        if verbose:
            print (
                "There are not enough DM peaks to be considered.\n" +
                "len(fhat['peaks_dens']) < accepted_peak_no"
                )
        return len(fhat["peaks_dens"])
    return accepted_peak_no


#  -stuff below this line are unstable but may be used if all else fails -----

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


def infer_stat_significant_threshold(sigma_no, fhat):
    """
    :sigma_no: float, how many sigma to use as the threshold
    :returns: threshold number for the density
    """
    return


def apply_density_threshold(total_peak_dens, fhat, threshold=0.9):
    """
    Make sure that the summed density of the DM peaks match the
    summed density of the galaxy KDE peaks.
    """
    peaks_mask = fhat["peaks_dens"] > threshold
    while(np.sum(fhat["peaks_dens"][peaks_mask]) < total_peak_dens):
        threshold -= .05
        peaks_mask = fhat["peaks_dens"] > threshold
    return threshold, np.sum(fhat["peaks_dens"][peaks_mask])



# def smooth_histograms(fhat):
#     smoothed = ndimage.filters.gaussian_filter(fhat["estimate"], sigma=3)
#
#     return
