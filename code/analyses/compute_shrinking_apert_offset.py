# ------------ read in df to be appended------------------
import pandas as pd
old_df = pd.read_hdf("../../data/offset_stat_129.h5", "df")

# ---------- import other libraries -------------
import numpy as np
import sys
# import time

sys.path.append("../")

import extract_catalog as ec
# following line only works when you are in the ks_KDE.r directory
import get_gal_centroids as cent
import compute_clst_prop as cp
# import cPickle
import sys
from get_gal_centroids import mag_to_lum


# ------------ read in hdf5 data files------------------
import h5py
h5File = "../../data/Illustris-1_fof_subhalo_myCompleteHaloCatalog_00135.hdf5"
f = h5py.File(h5File, "r")

# ------------ there are only 129 clst > 1e13 Msun -------------
assert len(sys.argv) > 1, "wrong number of arguments supplied\n" + \
    "usage: python compute_KDE_galDMoffset.py clstNo"  # suffix"
allClst = int(sys.argv[1])
file_suffix = '_{0}'.format(allClst) + '.h5'
print "examining {0} clusters in total".format(allClst)

# ------------ script parameters --------------------------------
compute_relaxedness0 = False

# weights to be used, DON"T CHANGE THE ORDER, last key should be i_lum
wts_keys = ["SubhaloMass", "SubhaloMassType4", "i_lum"]
wt_suffix = ["tot_mass", "stel_mass", "I_band_lum"]
wt_suffix = [wt + "_shrink" for wt in wt_suffix]


df_list = [ec.extract_clst(f, clstNo) for clstNo in range(allClst)]

cut_kwargs = {'DM_cut': 1e3, 'star_cut': 5e1}
for df in df_list:
    df[wts_keys[-1]] = np.abs(df["i_band"].apply(mag_to_lum))


results = {}
df_outlist = []
j = 0
file_suffix = "_" + wt_suffix[j] + '.pkl'

for j, wts_key in enumerate(wts_keys):
    # alternative syntax for line below is do a groupby using clusterNo then
    # use apply using lambda function
    offsets_list = \
        [cent.compute_shrinking_aperture_offset(df_list[i], f, i,
                                                cent.cut_reliable_galaxies,
                                                cut_kwargs,
                                                w=df_list[i][wts_key])
         for i in range(allClst)]

    # put offset list with different weights each become a col of data frame
    df_outlist.append(pd.DataFrame(offsets_list,
                                   columns=["offset_" + wt_suffix[j]]))


# cbind aka column bind
# df = pd.concat(df_outlist, axis=1)

if compute_relaxedness0:
    df["relaxedness1"] = [cp.compute_relaxedness0(df_list[i], f, i) for i in
                          range(allClst)]

f.close()
#---- Example code for appending to existing dataframe----------
# pd.concat([old_df, df], axis=1)

## append to hdf5
## f = open("offset_list" + file_suffix, 'w')
## cPickle.dump(offsets_list, f)
## f.close()
##
## f = open('relaxedness' + file_suffix, 'w')
## cPickle.dump(relaxedness_list, f)
