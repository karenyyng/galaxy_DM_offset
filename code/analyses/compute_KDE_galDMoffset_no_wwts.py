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
import pandas as pd

h5File = "../../data/Illustris-1_fof_subhalo_myCompleteHaloCatalog_00135.hdf5"
f = h5py.File(h5File, "r")

# there are only 129 clst > 1e13 Msun
assert len(sys.argv) > 1, "wrong number of arguments supplied\n" + \
    "usage: python compute_KDE_galDMoffset.py clstNo"  # suffix"
allClst = int(sys.argv[1])
# suffix = str(sys.argv[2])
compute_relaxedness0 = False

file_suffix = '_{0}_no_wwts'.format(allClst) + '.h5'

print "examining {0} clusters in total".format(allClst)

df_list = [ec.extract_clst(f, clstNo) for clstNo in range(allClst)]

cut_kwargs = {'DM_cut': 1e3, 'star_cut': 5e1}

results = {}

result_list = \
    [cent.compute_KDE_peak_offsets(df_list[i], f, i,
                                   cent.cut_reliable_galaxies,
                                   cut_kwargs)
     for i in range(allClst)]

# reformat the outputs and save to a dataframe
offsets_list = np.array([result_list[i][:2] for i in range(allClst)])

df = pd.DataFrame(offsets_list, columns=["no_wwts",
                                         "R200_no_wwts"])

fhat_list = [result_list[i][2] for i in range(allClst)]

# don't know what to do with the fhats for now
print "saving fhat with file suffix {0}".format(file_suffix)
out_f = open('../../data/fhat' + file_suffix, 'w')
pickle.dump(fhat_list, out_f)
out_f.close()

if compute_relaxedness0:
    df["relaxedness1"] = [cp.compute_relaxedness0(df_list[i], f, i) for i in
                          range(allClst)]

# old_df = pd.read_hdf
# store = pd.io.pytables.HDFStore(
#     "../../data/offset_stat" + file_suffix, mode="a")
# store.append(df, "df")
# store.close()

## append to hdf5
## f = open("offset_list" + file_suffix, 'w')
## pickle.dump(offsets_list, f)
## f.close()
##
## f = open('relaxedness' + file_suffix, 'w')
## pickle.dump(relaxedness_list, f)
f.close()
