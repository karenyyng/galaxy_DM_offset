"""unit tests for get gal centroids.py"""
import sys
sys.path.append("../")
from get_gal_centroids import *
from plot_gal_prop import *


#def test_get_py_peaks_and_density_weights():
#
#
#    return


if __name__=="__main__":
    # test_get_py_peaks_and_density_weights()
    # setup test data
    x = gaussian_mixture_data()  # random seed is set to 8192 by default
    res = do_KDE_and_get_peaks(x)
    peaks = res[0]
    fhat = convert_fhat_to_dict(res[1])

    get_py_peaks_and_density_weights(fhat)

    print fhat["peaks_coords"]
    print fhat["peaks_dens"]

    plot_KDE_peaks(fhat, allpeaks=True, showData=True)


