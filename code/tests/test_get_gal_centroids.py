"""unit tests for get gal centroids.py"""
import sys
sys.path.append("../")
from get_gal_centroids import *
from plot_gal_prop import *




def test_get_py_peaks_and_density_weights():
    res = do_KDE_and_get_peaks(x)
    fhat = convert_fhat_to_dict(res)

    get_py_peaks_and_density_weights(fhat)
    plot_KDE_peaks(fhat, allPeaks=True, showData=True)

    return


def test_weights_of_do_KDE_and_get_peaks():
    py_x = np.array(x)
    orig = np.array([-2, 2])
    dist_x = np.array([np.sqrt(np.dot(x_row - orig, x_row - orig))
                       for x_row in py_x])
    w = np.ones(len(np.array(x)))
    mask = dist_x < 0.3
    w[mask] = 50.  # weight points near the (-2, 2) peak more

    res = do_KDE_and_get_peaks(x, w=w)
    fhat = convert_fhat_to_dict(res)
    get_py_peaks_and_density_weights(fhat)
    plot_KDE_peaks(fhat, allPeaks=True, showData=True)
    return


def test_data1():
    fhat = {}
    fhat["estimate"] = np.array([[1, 1, 1, 1, 1, 1, 1],
                                 [1, 8, 3, 2, 1, 1, 1],
                                 [1, 7, 9, 3, 1, 1, 1],
                                 [3, 2, 4, 4, 5, 2, 1],
                                 [1, 1, 3, 2, 1, 1, 1],
                                 [1, 1, 1, 1, 2, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1]])

    a = np.arange(fhat["estimate"].shape[0])
    fhat["eval_points"] = np.vstack((a, a))
    return fhat


def test_find_peak_from_py_deriv():
    fhat = test_data1()
    # ans put in descending density
    correct_peaks_rowIx = [2, 3]
    correct_peaks_colIx = [2, 4]


    return


def test_bandwidth_matrix():
    return

if __name__ == "__main__":
    x = gaussian_mixture_data()  # random seed is set to 8192 by default
    # test_get_py_peaks_and_density_weights()
    # test_weights_of_do_KDE_and_get_peaks()
    pass
