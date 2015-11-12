"""
test code to ensure nothing breaks after modification
"""
from __future__ import division
import numpy as np
import sys
sys.path.append("../")
import extract_catalog as ec


# def test_wrap_coords_too_close_to_edge():
#     true_coord = np.array([
#
#     ])

def test_wrap_and_center_coord1():
    """this only tests the wrapping part"""
    true_coord = np.arange(-11, 11) * 1e3
    true_coord[0] = 0

    coord1 = 7.5e4 + true_coord[:11].copy()
    coord2 = true_coord[11:].copy()

    test_coord = np.concatenate((coord1, coord2))

    ans = ec.wrap_and_center_coord(test_coord, verbose=True)

    assert np.allclose(ans, true_coord), \
        "test1 {0} failed".format(ec.wrap_and_center_coord.__name__)

    return





# def test_wrap_and_center_coord2():
#     """test the wrapping THEN the centering, we treat the first particle as the
#     "center, i.e. subtract the coordinates of the first entry """
#     true_coord = np.arange(-11, 11) * 1e3
#     true_coord[0] = 1000
#
#     coord1 = 7.5e4 + true_coord[:11].copy()
#     coord2 = true_coord[11:].copy()
#
#     test_coord = np.concatenate((coord1, coord2))
#
#     ans = ec.wrap_and_center_coord(test_coord, verbose=True)
#
#     assert np.allclose(ans, true_coord), \
#         "test1 {0} failed".format(ec.wrap_and_center_coord.__name__)
#
#     return


def test_check_particle_parent_clst():
    test = np.hstack([np.ones(3),
                      np.ones(4) * 2,
                      np.ones(1) * 3])
    result = ec.check_particle_parent_clst(test, clsts=3,
                                           end_id=None)

    print(result)
    assert np.array_equal(np.array([3, 7]),
                          np.array(result))

    print("test pass")
    return result

if __name__ == "__main__":
    test_wrap_and_center_coord1()
    print "passed test1"
