"""
test code to ensure nothing breaks after modification
"""
from __future__ import division
import numpy as np
import sys
sys.path.append("../")
import extract_catalog as ec


def test_wrap_and_center_coord1():
    true_coord = np.arange(-11, 11) * 1e3
    true_coord[0] = 0

    coord1 = 7.5e4 + true_coord[:11].copy()
    coord2 = true_coord[11:].copy()

    test_coord = np.concatenate((coord1, coord2))
    #print "test_coord are \n {0}".format(test_coord)
    #print "true_coord are \n {0}".format(true_coord)

    ans = ec.wrap_and_center_coord(test_coord)

    assert np.allclose(ans, true_coord), \
        "test1 {0} failed".format(ec.wrap_and_center_coord.__name__)

    return


if __name__ == "__main__":
    test_wrap_and_center_coord1()
