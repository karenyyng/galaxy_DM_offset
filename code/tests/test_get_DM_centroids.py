import h5py
import sys
sys.path.append("../")

# import my own module to be tested
from get_DM_centroids import *


def test_construct_h5_file_for_saving_fhat():
    import os
    from collections import OrderedDict
    metadata = OrderedDict({})
    metadata["clstNo"] = np.arange(1, 3)
    metadata["cut"] = {"min": "placeholder"}
    metadata["weights"] = {"i_band": "placeholder"}
    metadata["los_axis"] = [1]
    metadata["xi"] = ["0.", "3.14"]
    metadata["phi"] = ["0.", "3.14"]
    metadata["sig_fraction"] = ["0.1", "0.15", "0.2"]
    metadata["kernel_width"] = ["0.", "25"]

    filename = "test.h5"
    output_path = "./"
    construct_h5_file_for_saving_fhat(metadata, filename,
                                      output_path=output_path)

    test_h5 = h5py.File(output_path + filename)
    path = "2"
    assert test_h5[path].attrs["info"] == "clstNo", \
        "problem saving metadata correctly for clstNo"

    path += "/min"
    assert test_h5[path].attrs["info"] == "cut", \
        "problem saving metadata correctly for cut"

    path += "/i_band"
    assert test_h5[path].attrs["info"] == "weights", \
        "problem saving metadata correctly for weights"

    path += "/1"
    assert test_h5[path].attrs["info"] == "los_axis", \
        "problem saving metadata correctly for los_axis"

    path += "/3.14"
    assert test_h5[path].attrs["info"] == "xi", \
        "problem saving metadata correctly for xi"

    path += "/3.14"
    assert test_h5[path].attrs["info"] == "phi", \
        "problem saving metadata correctly for phi"

    path += "/0.2"
    assert test_h5[path].attrs["info"] == "sig_fraction", \
        "problem saving metadata correctly for sig_fraction"

    path += "/25"
    assert test_h5[path].attrs["info"] == "kernel_width", \
        "problem saving metadata correctly for kernel_width"


    test_h5.close()
    os.system("rm ./test.h5")

    return
