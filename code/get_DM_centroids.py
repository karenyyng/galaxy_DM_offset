"""for inferring DM centroids / peaks """
from __future__ import (print_function, division)
import numpy as np
import matplotlib.pyplot as plt
import get_KDE


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
