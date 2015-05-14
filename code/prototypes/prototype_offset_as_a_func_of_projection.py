""" prototype script for final run """
from __future__ import (print_function, unicode_literals,
                        division, absolute_import)
import h5py
# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import sys
sys.path.append("../")

import extract_catalog as ext_cat
import get_gal_centroids as getg
from collections import OrderedDict

dataPath = "../../data/"

original_f = h5py.File(dataPath +
                "Illustris-1_fof_subhalo_myCompleteHaloCatalog_00135.hdf5")

# ================ make all decisions ===========================
clstNos = [4, 5]
cut_kwargs={"DM_cut": 1e3, "star_cut": 1e2}
# nsides of HEALpix are powers of 2
nsides = 2
xi_array, phi_array = getg.angles_given_HEALpix_nsides(2)
wts = None


# start storing meta data based on decision on what to run
metadata = OrderedDict({})
metadata["clstNo"] = clstNo
metadata["cut"] = "min"
metadata["weights"] = "None"

# ================ make all decisions ===========================

# if w is None:
#   metadata["weights"] = "None"
# elif weight is None:
#     raise ValueError("for non-trivial weights, " +
#                     "weight needs to be specified")
# else:

df = ext_cat.extract_clst(original_f, clstNo)
col = [k for k in df.keys() if "SubhaloPos" in k]

mask = getg.cut_reliable_galaxies(df, **cut_kwargs)
data = np.array(df[col][mask])

# # Start projection!
data_arr, xi_array, phi_array, los_axis = \
    getg.prep_data_with_cuts_and_proj(df, getg.cut_reliable_galaxies,
                                      cut_kwargs, nside_pow=1,
                                      verbose=True)

proj_no = 2
stuff = map(lambda d, xi, phi:
            getg.compute_KDE_peak_offsets(d, original_f, clstNo, xi=xi,
                                          phi=phi, los_axis=los_axis),
            data_arr[:proj_no], xi_array[:proj_no], phi_array[:proj_no])

offset_list = [s[0] for s in stuff]
offset_R200_list = [s[1] for s in stuff]
fhat_list = [s[2] for s in stuff]

# compile list of meta data for one cluster!


peak_df = getg.convert_dict_peaks_to_df(fhat_list, wt="None")

# getg.convert_dict_dens_to_h5(fhat_list, wt="None")

