"""Prototype script for final run.
"""
from __future__ import (print_function,
                        division, absolute_import)
import logging
import pandas as pd
import os

from datetime import datetime
datetime_stamp = datetime.now().strftime("%D").replace('/', '_')

# =========== Decide what to output ============================

dataPath = "../../data/"


# these are for adjusting the number of clusters that are analyzed
clstID_h5filepath = dataPath + "rich_cluster_ID.h5"
total_clstNo = 43
# start_clstNo = 13


logging_filename = "star_logging_{0}_{1}.log".format(
    total_clstNo, datetime_stamp)
# output fhat

output_keys = [
    # 'eval_points',
    # "estimate",
    # "bandwidth_matrix_H",
    "shrink_cent",
    # "centroid",
    # 'BCG',
    "peaks_dens",
    "peaks_xcoords",
    "peaks_ycoords"
]

# =========== Decide what to output ============================
assert total_clstNo <=128 and total_clstNo > 0, \
    "0 < total_clstNo <= 128"

output_fhat_filename = \
    "stars_fhat_clst_no_density_{0}_{1}.h5".format(
        total_clstNo, datetime_stamp)
logging.basicConfig(filename=logging_filename, level=logging.INFO)

from collections import OrderedDict
import numpy as np
import sys
sys.path.append("../")

import extract_catalog as ext_cat
import get_gal_centroids as getg
import get_KDE as KDE
import calculate_astrophy_quantities as cal_astro

verbose = True
# Do not import h5py before opening the store, it may crash
import h5py
original_f = h5py.File(dataPath +
                       "Illustris-1_fof_subhalo_myCompleteHaloCatalog_00135" +
                       ".hdf5", "r")

clstID_h5file = h5py.File(clstID_h5filepath, "r")


# ================ make science related decisions ===========================

pos_cols = ["SubhaloPos{0}".format(i) for i in range(3)]
metadata = OrderedDict({})

# no. of clsters - want these as strings, not int!
# use the last n-th cluster instead of the first n-th because they
# contain less particles and run faster
metadata["clstNo"] = \
    clstID_h5file["rich_cluster_ID"][-total_clstNo:]
    # [str(i) for i in range(start_clstNo, start_clstNo + total_clstNo)]
    #  range(129)

### cuts
# cut_kwargs = {"DM_cut": 1e3, "star_cut": 5e2}
# cut_methods = {"min": getg.cut_reliable_galaxies}  # use cut_dim_galaxies!?
# cut_cols = {"min": pos_cols}
# metadata["cut"] = {"min": cut_kwargs}

assumed_z = 0.3
cut_kwargs = {"limiting_mag_band": "apparent_i_band",
              "limiting_mag": 24.4
              }

cut_methods = {"mag": getg.cut_dim_galaxies}
cut_cols = {"mag": "apparent_i_band"}
metadata["cut"] = {"mag": cut_kwargs}

# weights
# metadata["weights"] = {
#     "i_band": getg.mag_to_lum
#     }
metadata["weights"] = {
    "None": None
}

# projections
metadata["los_axis"] = [str(1)]  # use y-axis as los axis
# metadata["los_axis"] = [str(2)]  # use z-axis as los axis

# Want to use string as key, not floats!
nside = 8  # nsides of HEALpix are powers of 2, pix for 16 nsides = 3072 / 2
phi_arr, xi_arr = \
    getg.angles_given_HEALpix_nsides(nside, return_all=True)
# phi_arr, xi_arr = [[0.0], [0.0]]

xi_arr = ['{0:1.10f}'.format(xi) for xi in xi_arr]
phi_arr = ['{0:1.10f}'.format(phi) for phi in phi_arr]
metadata['projection'] = zip(xi_arr, phi_arr)

logging.info (
    "{} projections per cluster are constructed".format(
        len(metadata["projection"])))

# ============== set up output file structure  ===========
# check_metadata against illegal types
# create HDF5 file structure first!
if os.path.isfile(dataPath + output_fhat_filename):
    os.remove(dataPath + output_fhat_filename)
h5_fstream = \
    getg.construct_h5_file_for_saving_fhat(
        metadata, output_fhat_filename, output_path=dataPath)

# ============== prepare data based on the metadata ===========
clst_metadata = OrderedDict({})
for clstNo in metadata["clstNo"]:
    logging.info("Processing clst {0} ".format(int(clstNo)) +
                 "out of the range {0} to {1}".format(
                     metadata['clstNo'][-total_clstNo:][0],
                     metadata['clstNo'][-1]))
    peak_df = pd.DataFrame()
    clst_metadata["clstNo"] = clstNo
    df = ext_cat.extract_clst(original_f, clstNo)

    illustris_cosmo = cal_astro.get_Illustris_cosmology()
    abs_mag = '_'.join(cut_kwargs['limiting_mag_band'].split('_')[1:])
    df[cut_kwargs['limiting_mag_band']] = \
        cal_astro.convert_abs_mag_to_apparent_mag(
         df[abs_mag], illustris_cosmo, z=assumed_z)

    dfs_with_cuts, richness = \
        getg.prep_data_with_cuts_and_wts(
            df, metadata["cut"], cut_methods, cut_cols,
            metadata["weights"], verbose)

    for cut, thisdf in dfs_with_cuts.iteritems():
        clst_metadata["cut"] = cut

        for wt_key in metadata["weights"]:
            clst_metadata["weights"] = wt_key
            weights = thisdf[wt_key + "_wt"]

            # save the richness
            h5_fstream[
                str(clstNo) + '/' + cut + '/' + wt_key + '/'
                + "richness"
            ] = richness[cut]

            for los_axis in metadata["los_axis"]:
                clst_metadata["los_axis"] = los_axis

                for i in range(len(metadata["projection"])):
                    clst_metadata["projection"] = metadata["projection"][i]

                    data = getg.project_coords(np.array(thisdf[pos_cols]),
                                               clst_metadata["projection"][0],
                                               clst_metadata["projection"][1],
                                               los_axis=los_axis)

                    cols = np.arange(data.shape[1]) != int(los_axis)
                    data = data[:, cols]

                    fhat = KDE.do_KDE_and_get_peaks(data, weights)

                    # store shrinking aperture info in fhat
                    fhat['shrink_cent'] = getg.shrinking_apert(
                        data, w=weights
                    )

                    fhat['centroid'] = getg.compute_weighted_centroids(
                        data, w=weights
                    )

                    # get the BCG in a particular band after projecting their
                    thisdf['spat1'] = data[:, cols[0]]
                    thisdf['spat2'] = data[:, cols[1]]
                    fhat['BCG'] = getg.get_BCG_location(
                        thisdf, band="i_band",
                        spat_key1='spat1', spat_key2='spat2'
                    )

                    getg.convert_dict_dens_to_h5(
                        fhat, clst_metadata,
                        h5_fstream, fixed_size_data_keys=output_keys)

h5_fstream.close()
