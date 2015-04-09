import h5py
import numpy as np
import sys
# import time

sys.path.append("../")

import extract_catalog as ec
# following line only works when you are in the ks_KDE.r directory
import get_gal_centroids as cent
import compute_clst_prop as cp
import pickle
import sys
from test_peak_methods import mag_to_lum
import pandas as pd

h5File = "../../data/Illustris-1_fof_subhalo_myCompleteHaloCatalog_00135.hdf5"
f = h5py.File(h5File, "r")

# there are only 129 clst > 1e13 Msun
assert len(sys.argv) > 1, "wrong number of arguments supplied\n" + \
    "usage: python compute_KDE_galDMoffset.py clstNo"  # suffix"
allClst = int(sys.argv[1])
compute_relaxedness0 = False

# wts_key = ["SubhaloMass", "SubhaloMassType4", "i_band",
wts_key = ["i_lum"]
wt_suffix = ["I_lum"]  # ["tot_mass", "stel_mass", "I_band_lum"]

file_suffix = '_{0}'.format(allClst) + '.h5'
print "examining {0} clusters in total".format(allClst)

df_list = [ec.extract_clst(f, clstNo) for clstNo in range(allClst)]

cut_kwargs = {'DM_cut': 1e3, 'star_cut': 5e1}

for df in df_list:
    df[wts_key[0]] = np.abs(df["i_band"].apply(mag_to_lum))

results = {}

df_outlist = []
j = 0
file_suffix = "_" + wt_suffix[j] + '.pkl'
result_list = \
    [cent.compute_KDE_peak_offsets(df_list[i], f, i,
                                   cent.cut_reliable_galaxies,
                                   cut_kwargs,
                                   w=df_list[i][wts_key[j]])
     for i in range(allClst)]

# reformat the outputs and save to a dataframe
offsets_list = np.array([result_list[i][:2] for i in range(allClst)])

df_outlist.append(pd.DataFrame(offsets_list,
                               columns=["offset_" + wt_suffix[j],
                                        "offset_R200_" + wt_suffix[j]]))

fhat_list = [result_list[i][2] for i in range(allClst)]

# don't know what to do with the fhats for now
print "saving fhat with file suffix {0}".format(file_suffix)
out_f = open('../../data/fhat' + file_suffix, 'w')
pickle.dump(fhat_list, out_f)
out_f.close()

if compute_relaxedness0:
    df["relaxedness1"] = [cp.compute_relaxedness0(df_list[i], f, i) for i in
                          range(allClst)]

f.close()
#---- Example code for appending to existing dataframe----------
# old_df = pd.read_hdf("../../data/offset_stat_129.h5", "w")
# pd.concat([old_df, df], axis=1)

## append to hdf5
## f = open("offset_list" + file_suffix, 'w')
## pickle.dump(offsets_list, f)
## f.close()
##
## f = open('relaxedness' + file_suffix, 'w')
## pickle.dump(relaxedness_list, f)
