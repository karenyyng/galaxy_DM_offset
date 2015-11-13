"""
This contains module for computing distances between fhat_star & fhat (from
DM).
"""
import numpy as np
import sys
sys.path.append("../")
import get_DM_centroids as getDM


def compute_distance_between_DM_and_gal_peaks(
        fhat_star, fhat, fhat_star_to_DM_coord_conversion=1. / 0.704):
    """
    Parameters
    ===========
    fhat_star: one of the fhat_stars from `get_KDE`
        coordinates from fhat_star is in kpc / h,
        convert this to kpc by multiplying fhat_star_coordinates
    fhat: fhat output from `getDM.make_histogram_with_some_resolution`

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
    fhat_star["dist"] = dist
    fhat_star["DM_ixes"] = DM_ixes
    fhat_star["gal_peak_no"] = gal_peak_no
    fhat_star["DM_peak_no"] = DM_peak_no

    # We use Euclidean distance for our query, i.e. p=2.
    return (fhat_star["dist"], fhat_star["DM_ixes"]), fhat_star["gal_peak_no"],\
        fhat_star["DM_peak_no"]
