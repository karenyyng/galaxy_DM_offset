"""Prototype script for final run.
"""
from __future__ import (print_function,
                        division, absolute_import)
import pandas as pd
import os

from datetime import datetime
datetime_stamp = datetime.now().strftime("%D").replace('/', '_')

# =========== Decide what to output ============================
dataPath = "../../data/"
total_clstNo =  10

# =========== Decide what to output ============================
assert total_clstNo <=128 and total_clstNo >= 0, \
    "0 <= total_clstNo <= 128"
output_fhat_filename = \
    "test_stars_fhat_clst{0}_{1}.h5".format(total_clstNo, datetime_stamp)
StoreFile = \
    "test_stars_peak_df_clst{0}_{1}.h5".format(total_clstNo, datetime_stamp)
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
# Do not import h5py before opening the store, it may crash
import h5py
original_f = h5py.File(dataPath +
                       "Illustris-1_fof_subhalo_myCompleteHaloCatalog_00135" +
                       ".hdf5")

# ================ make science related decisions ===========================

pos_cols = ["SubhaloPos{0}".format(i) for i in range(3)]
metadata = OrderedDict({})

# no. of clsters - want these as strings, not int!
metadata["clstNo"] = [str(i) for i in range(28, 38)]  #  range(129)

# cuts
cut_kwargs = {"DM_cut": 1e3, "star_cut": 5e2}
cut_methods = {"min": getg.cut_reliable_galaxies}
cut_cols = {"min": pos_cols}
metadata["cut"] = {"min": cut_kwargs}

# weights
metadata["weights"] = OrderedDict({
    "i_band": getg.mag_to_lum
    })

# projections
nside = 1  # nsides of HEALpix are powers of 2, pix for 16 nsides = 3072 / 2
metadata["los_axis"] = [str(1)]  # use z-axis as los axis

# Want to use string as key, not floats!
metadata["xi"], metadata["phi"] = getg.angles_given_HEALpix_nsides(nside)
metadata["xi"] = ['{0:1.10f}'.format(xi) for xi in metadata["xi"]]
metadata["phi"] = ['{0:1.10f}'.format(phi) for phi in metadata["phi"]]

print (
    "{} projections per cluster are constructed".format(len(metadata["xi"])))

# ============== set up output file structure  ===========
# check_metadata against illegal types
# create HDF5 file structure first!
if os.path.isfile(dataPath + output_fhat_filename):
    os.remove(dataPath + output_fhat_filename)
h5_fstream = \
    getg.construct_h5_file_for_saving_fhat(metadata,
                                           output_fhat_filename,
                                           output_path=dataPath
                                           )


# ============== prepare data based on the metadata ===========
clst_metadata = OrderedDict({})
for clstNo in metadata["clstNo"]:
    print ("processing clst {0} ".format(int(clstNo) + 1) +
           "out of {0}".format(len(metadata["clstNo"])))
    peak_df = pd.DataFrame()
    clst_metadata["clstNo"] = clstNo
    df = ext_cat.extract_clst(original_f, clstNo)

    dfs_with_cuts, richness = \
        getg.prep_data_with_cuts_and_wts(df, metadata["cut"],
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
                                               los_axis=los_axis)

                    cols = np.arange(data.shape[1]) != int(los_axis)
                    data = data[:, cols]

                    fhat = KDE.do_KDE_and_get_peaks(data, weights)
                    # This is not needed since the offset will be computed
                    # w.r.t. dark matter peak instead
                    # getg.compute_KDE_peak_offsets(fhat, original_f,
                    #                               clst_metadata["clstNo"])
                    peak_df = \
                        getg.convert_dict_peaks_to_df(fhat, clst_metadata)
                    store.append("peak_df", peak_df)

                    # clst_metadata[cut + 'richness'] = richness
                    getg.convert_dict_dens_to_h5(fhat, clst_metadata,
                                                 h5_fstream)

store.close()
