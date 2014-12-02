"""
test code to ensure nothing breaks after modification
"""
from __future__ import division
import numpy as np
import sys
sys.path.append("../")
import extract_catalog as ec


def test_wrap_and_center_coord1():
    true_coord = np.arange(-10, 11) * 1e3
    coord1 = 7.5e4 + true_coord[:10].copy()
    coord2 = true_coord[10:].copy()

    test_coord = np.concatenate((coord1, coord2))

    #print "test_coord are \n {0}".format(test_coord)
    #print "true_coord are \n {0}".format(true_coord)
    #print "result from function = \n {0}".format(ans)

    ans = ec.wrap_and_center_coord(test_coord)

    assert np.allclose(ans, true_coord), \
        "test1 {0} failed".format(ec.wrap_and_center_coord.__name__)

    return


def test_wrap_and_center_coord2():
    """ test 2 - coordinates of test array should not be changed """
    test_coord = np.arange(1, 10) + 7.5e4 / 2.
    ans = ec.wrap_and_center_coord(test_coord)

    assert np.array_equal(ans, test_coord - np.median(test_coord)), \
        "test2 {0} failed".format(ec.wrap_and_center_coord.__name__)

    return


def test_fix_clst_cat():
    #assert
    return

if __name__ == "__main__":
    test_wrap_and_center_coord1()
    test_wrap_and_center_coord2()
