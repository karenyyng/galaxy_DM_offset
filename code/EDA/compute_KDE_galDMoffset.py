import h5py
import numpy as np
import sys
import time

sys.path.append("../")

import extract_catalog as ec
# following line only works when you are in the ks_KDE.r directory
import get_gal_centroids as cent
import compute_clst_prop as cp
import pickle
import sys
import argparse
import pandas as pd

h5File = "../../data/Illustris-1_fof_subhalo_myCompleteHaloCatalog_00135.hdf5"
f = h5py.File(h5File, "r")
# parser = argparse.ArgumentParser(description="Compute offsets")
# parser.add_argument("no of clsters to process", metavar='allClst', type=int,
#                     help="")

# there are only 129 clst > 1e13 Msun
assert len(sys.argv) > 2, "wrong number of arguments supplied\n" + \
    "usage: python compute_KDE_galDMoffset.py clstNo suffix"
allClst = int(sys.argv[1])
suffix = str(sys.argv[2])

print "examining {0} clusters in total".format(allClst)
file_suffix = '_{0}'.format(allClst) + suffix + '.h5'
print "file suffix is {0}".format(file_suffix)

df_list = [ec.extract_clst(f, clstNo) for clstNo in range(allClst)]

cut_kwargs = {'DM_cut': 1e3, 'star_cut': 5e1}

result_list = [cent.compute_KDE_peak_offsets(df_list[i], f, i,
                                             cent.cut_reliable_galaxies,
                                             cut_kwargs,
                                             w=df_list[i]["SubhaloMass"])
               for i in range(allClst)]

# reformat the outputs and save to a dataframe
offsets_list = np.array([result_list[i][:2] for i in range(allClst)])

df = pd.DataFrame(offsets_list, columns=["offset", "offset_R200"])
fhat_list = [result_list[i][2] for i in range(allClst)]

df["relaxedness1"] = [cp.compute_relaxedness0(df_list[i], f, i) for i in
                      range(allClst)]

store = pd.io.pytables.HDFStore("../../data/offset_stat" + file_suffix, mode="w")
store.append("df", df)
store.close()

# don't know what to do with the fhats for now
f = open('../../data/fhat' + file_suffix, 'w')
pickle.dump(fhat_list, f)
f.close()

# store.append("df/fhat_list", fhat_list)


# append to hdf5
# f = open("offset_list" + file_suffix, 'w')
# pickle.dump(offsets_list, f)
# f.close()
#
# f = open('relaxedness' + file_suffix, 'w')
# pickle.dump(relaxedness_list, f)
# f.close()
#
