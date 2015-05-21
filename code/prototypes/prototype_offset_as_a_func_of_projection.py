""" prototype script for final run """
from __future__ import (print_function,
                        division, absolute_import)
import pandas as pd
dataPath = "../../data/"
store = pd.HDFStore(dataPath + "test_framework.h5")
import numpy as np
# import matplotlib.pyplot as plt
import sys
sys.path.append("../")

import extract_catalog as ext_cat
import get_gal_centroids as getg
import get_KDE as KDE
from collections import OrderedDict

verbose = True
import h5py
original_f = h5py.File(dataPath +
                       "Illustris-1_fof_subhalo_myCompleteHaloCatalog_00135.hdf5")

# ================ make all decisions ===========================
pos_cols = ["SubhaloPos{0}".format(i) for i in range(3)]
metadata = OrderedDict({})
metadata["clstNo"] = [4]  # , 5]

cut_kwargs = {"DM_cut": 1e3, "star_cut": 5e2}
cut_methods = {"min": getg.cut_reliable_galaxies}
cut_cols = {"min": pos_cols}
metadata["cuts"] = {"min": cut_kwargs}

nside = 1
metadata["los_axis"] = 2  # use z-axis as los axis
# nsides of HEALpix are powers of 2
metadata["xi"], metadata["phi"] = getg.angles_given_HEALpix_nsides(nside)

metadata["weights"] = OrderedDict({"no": None,
                                   "i_band": getg.mag_to_lum})

# check_metadata
# create HDF5 file structure first!

# ============== prepare data based on the metadata ===========
clst_metadata = {}
list_peak_df = []
for clstNo in metadata["clstNo"]:
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

            for i in range(1):  # range(len(metadata["xi"])):
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







    # for df in dfs_with_cuts.values():
    #     proj, xi_array, phi_array = \
    #         getg.project_cluster_df(df[pos_cols], metadata["nside_pow"],
    #                                 metadata["los_axis"], verbose)

    # do_KDE_and_get_peaks()
    # stuff = map(lambda d, xi, phi:
    #             getg.compute_KDE_peak_offsets(d, original_f, clstNo, xi=xi,
    #                                           phi=phi, los_axis=los_axis),
    #             data_arr[:proj_no], xi_array[:proj_no], phi_array[:proj_no])
#
# offset_list = [s[0] for s in stuff]
# offset_R200_list = [s[1] for s in stuff]
# fhat_list = [s[2] for s in stuff]
#
# # compile list of meta data for one cluster!
#
#
# peak_df = getg.convert_dict_peaks_to_df(fhat_list, wt="None")

# getg.convert_dict_dens_to_h5(fhat_list, wt="None")
store.close()
