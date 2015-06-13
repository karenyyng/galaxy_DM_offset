"""Prototype script for final run."""
from __future__ import (print_function,
                        division, absolute_import)
import pandas as pd
import os
import datetime

timestamp = str(datetime.datetime.now())[:-7]
dataPath = "../../data/"
output_fhat_path = "test_DM_KDE_fhat_129_" + timestamp + ".h5"
StoreFile = "test_DM_peak_129_" + timestamp + ".h5"
if os.path.isfile(dataPath + StoreFile):
    os.remove(dataPath + StoreFile)
store = pd.HDFStore(dataPath + StoreFile)

import numpy as np
import sys
sys.path.append("../")

import extract_catalog as ext_cat
import get_gal_centroids as getg
import get_KDE as KDE
from collections import OrderedDict

verbose = True
import h5py
original_f = h5py.File(dataPath +
                       "Illustris-1_fof_subhalo_myCompleteHaloCatalog_00135" +
                       ".hdf5")

print ("Writing out to data to {0} and \n".format(output_fhat_path) +
       "{0}".format(StoreFile))
# ================ make all decisions ===========================

pos_cols = ["SubhaloPos{0}".format(i) for i in range(3)]
metadata = OrderedDict({})

# no. of clsters
metadata["clstNo"] = range(129)  # range(129)  # range(20)  # [4]  # [5]

# cuts
cut_kwargs = {"DM_cut": 1e3}
cut_methods = {"no": None}
cut_cols = {"no": pos_cols}
metadata["cuts"] = {"no": None}

# weights
metadata["weights"] = OrderedDict({
    # "no": None,
    # "i_band": getg.mag_to_lum
    "SubhaloMass": getg.get_subhalo_mass
    })

# projections
nside = 16  # nsides of HEALpix are powers of 2, pix for 16 nsides = 3072 / 2
metadata["los_axis"] = [2]  # use z-axis as los axis
metadata["xi"], metadata["phi"] = ([0], [0])  # getg.angles_given_HEALpix_nsides(nside)

# ============== set up output file structure  ===========
# check_metadata against illegal types
# create HDF5 file structure first!
if os.path.isfile(dataPath + output_fhat_path):
    os.remove(dataPath + output_fhat_path)
h5_fstream = getg.construct_h5_file_for_saving_fhat(metadata, dataPath,
                                                    output_fhat_path)

print ("no of clusters to process : {}".format(len(metadata["clstNo"])))
# ============== prepare data based on the metadata ===========
clst_metadata = OrderedDict({})
for clstNo in metadata["clstNo"]:
    print ("processing clst {0} ".format(clstNo) +
           "out of {0}".format(len(metadata["clstNo"])))
    peak_df = pd.DataFrame()
    clst_metadata["clstNo"] = clstNo
    df = ext_cat.extract_clst(original_f, clstNo)

    dfs_with_cuts = \
        getg.prep_data_with_cuts_and_wts(df, metadata["cuts"],
                                         cut_methods, cut_cols,
                                         metadata["weights"],
                                         verbose)

    for cut, thisdf in dfs_with_cuts.iteritems():
        clst_metadata["cut"] = cut

        for wt_key in metadata["weights"]:
            clst_metadata["weights"] = wt_key
            weights = thisdf[wt_key + "_wt"]

            for los_axis in metadata["los_axis"]:
                clst_metadata["los_axis"] = los_axis

                for i in range(len(metadata["xi"])):
                    clst_metadata["xi"] = metadata["xi"][i]
                    clst_metadata["phi"] = metadata["phi"][i]

                    data = getg.project_coords(np.array(thisdf[pos_cols]),
                                               clst_metadata["xi"],
                                               clst_metadata["phi"],
                                               los_axis=metadata["los_axis"])

                    col = np.arange(data.shape[1]) != metadata["los_axis"]
                    data = data[:, col]

                    fhat = KDE.do_KDE_and_get_peaks(data, weights)
                    # this is not needed since the offset will be computed
                    # w.r.t. dark matter peak instead
                    # getg.compute_KDE_peak_offsets(fhat, original_f,
                    #                               clst_metadata["clstNo"])
                    peak_df = \
                        getg.convert_dict_peaks_to_df(fhat, clst_metadata)
                    store.append("peak_df", peak_df)

                    getg.convert_dict_dens_to_h5(fhat, clst_metadata,
                                                 h5_fstream)

store.close()
