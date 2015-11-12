import cPickle
import sys
import h5py
sys.path.append("../")

dataPath = "../../data/fig2_data/"
f = h5py.File(dataPath + "compare_methods.h5", compression="gzip",
              complevel=9, mode="a")

# list all the layers of iterations
data_size = [20, 50, 100, 500]
data = ["dumb", "gauss", "bimodal"]
methods = ["cent", "KDE", "shrink"]

def fhats_to_h5():
    """
    needs refactoring
    should grab the keys of fhat from the pickle files,
    then recursively iterate through keys at each level to write out to h5 file 
    stream
    """
    for size in data_size:
        try:
            gp = f.create_group(str(size))
        except ValueError:
            gp = f[str(size)]

        for dkey in data:
            try:
                subgrp = gp.create_group(dkey)
            except ValueError:
                subgrp = gp[dkey]

            if dkey != "dumb":
                for mtd in methods:
                        try:
                            subgrp1 = subgrp.create_group(mtd)
                        except ValueError:
                            subgrp1 = subgrp[mtd]

                        temp_fhat = cPickle.load(
                            open(dataPath + mtd + "_" + dkey + str(size) + ".pkl"))

                        for k in temp_fhat:
                            subgrp1[k] = temp_fhat[k]
            else:
                for mtd in ["cent", "KDE1", "KDE2", "shrink"]:
                        try:
                            subgrp1 = subgrp.create_group(mtd)
                        except ValueError:
                            subgrp1 = subgrp[mtd]

                        temp_fhat = cPickle.load(
                            open(dataPath + mtd + "_" + dkey + str(size) + ".pkl"))

                        for k in temp_fhat:
                            subgrp1[k] = temp_fhat[k]

    return

def mock_data_to_h5():
    for size in data_size:
        try:
            gp = f.create_group(str(size))
        except ValueError:
            gp = f[str(size)]

        for dkey in data:
            try:
                subgrp = gp.create_group(dkey)
            except ValueError:
                subgrp = gp[dkey]

            subgrp["data"] = cPickle.load(
                open(dataPath + "data/" + dkey + str(size) + ".pkl", "r"))

fhats_to_h5()
mock_data_to_h5()
f.close()
