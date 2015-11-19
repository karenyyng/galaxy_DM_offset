"""Prototype script for final run.
Copied from `fhat_star_as_a_func_of_projection.py`

This
* reads the metadata for the cuts from the fhat_star*.h5 file
* produces the DM projections accordingly
"""
from __future__ import (print_function,
                        division, absolute_import)
import pandas as pd
import os
from scipy import ndimage
from datetime import datetime
datetime_stamp = datetime.now().strftime("%D").replace('/', '_')

data_path = "../../data/"
# ------- specify output file paths  -----------------------
print ("Current date is {}".format(datetime_stamp))
total_clstNo = 50
output_fhat_filename = \
    "test_DM_fhat_clst{0}_{1}.h5".format(total_clstNo, datetime_stamp)
StoreFile = \
    "test_DM_peaks_df_clst{0}_{1}.h5".format(total_clstNo, datetime_stamp)
if os.path.isfile(data_path + StoreFile):
    os.remove(data_path + StoreFile)
store = pd.HDFStore(data_path + StoreFile)

print ("Outputing files to:")
print (output_fhat_filename)
print (StoreFile)

import numpy as np
import sys
from collections import OrderedDict
sys.path.append("../")

import extract_catalog as ec
import get_DM_centroids as getDM
import get_gal_centroids as getgal

# import get_KDE as KDE
# import compute_distance as getDist

verbose = True
import h5py
DM_h5file = data_path + \
    "Illustris-1_00135_APillepich_KarenNG_ParticleData_Group_PartType1.h5"

DM_fstream = h5py.File(DM_h5file)
# ================ make all decisions ===========================
# Specify fhat_star input file paths
input_datetime_stamp = '11_18_15'
input_star_file = \
    "test_stars_peak_df_clst{}_{}.h5".format(total_clstNo, input_datetime_stamp)
input_h5_key = "peak_df"
DM_metadata, star_peak_df = \
    getDM.retrieve_DM_metadata_from_gal_metadata(
        data_path, input_star_file, input_h5_key)

star_gpBy, star_gpBy_keys = \
    getgal.get_clst_gpBy_from_DM_metadata(star_peak_df)

DM_metadata["kernel_width"] = [0, 3, 30 * 4]
sig_fraction = 0.2

# ============== set up output file structure  ===========

# Check_metadata against illegal types
# Create HDF5 file structure first!
if os.path.isfile(data_path + output_fhat_filename):
    os.remove(data_path + output_fhat_filename)

print ("{} projections per cluster are constructed.".format(
    len(DM_metadata["xi"])))

h5_fstream = \
    getDM.construct_h5_file_for_saving_fhat(DM_metadata,
                                            output_fhat_filename,
                                            output_path=data_path
                                            )

# ============== prepare data based on the metadata ===========
pos_cols = ["SubhaloPos{}".format(i) for i in range(3)]

clst_metadata = OrderedDict({})
#### CHANGE THE RANGE of line below
for clstNo in DM_metadata["clstNo"]:
    print ("Processing clst {0} ".format(int(clstNo) + 1) +
           "out of {0}".format(len(DM_metadata["clstNo"])))
    peak_df = pd.DataFrame()
    clst_metadata["clstNo"] = clstNo  # clstNo is a string
    clstNo = int(clstNo)
    coord_dict = \
        ec.get_DM_particles([clstNo], DM_fstream, dataPath=data_path)[clstNo]

    # There are no cuts / weights for the DM particles.
    # However, the DM peaks are informed by the galaxy peaks.
    # The choice of cuts and weights affect the galaxy peaks.
    for cut in DM_metadata["cut"]:
        clst_metadata['cut'] = cut

        for weights in DM_metadata["weights"]:
            clst_metadata['weights'] = weights

            for los_axis in DM_metadata["los_axis"]:
                clst_metadata["los_axis"] = los_axis
                los_axis = int(los_axis)

                #### CHANGE THE RANGE of line below
                for i in range(len(DM_metadata["xi"])):
                    clst_metadata["xi"] = DM_metadata["xi"][i]
                    clst_metadata["phi"] = DM_metadata["phi"][i]

                    data = getgal.project_coords(
                        coord_dict["coords"], clst_metadata["xi"],
                        clst_metadata["phi"],
                        los_axis=clst_metadata["los_axis"])

                    col = np.arange(data.shape[1]) != los_axis
                    data = data[:, col]

                    #### CHANGE THE RANGE of DM_metadata["kernel_width"] below
                    # for kernel_width in DM_metadata["kernel_width"]:
                    for kernel_width in DM_metadata["kernel_width"]:

                        # Find the correpsonding galaxy projection
                        gpBy_keys = \
                            tuple([clst_metadata[k]
                                   for k in star_gpBy_keys])
                        print ("gpBy_keys = ", gpBy_keys)
                        print ("len(gpBy_keys) = ", len(gpBy_keys))
                        fhat_stars = \
                            star_gpBy.get_group(gpBy_keys)

                        # Save the cluster metadata as strings
                        # These are categorical.
                        clst_metadata["kernel_width"] = \
                            '{0:0.0f}'.format(kernel_width)
                        clst_metadata["sig_fraction"] = \
                            '{0:0.2f}'.format(sig_fraction)

                        # Can think of directly smoothing the histogrammed
                        # data from kernel_width = 0  here to reduce
                        # computation time.
                        fhat = \
                            getDM.make_histogram_with_some_resolution(
                                coord_dict, resolution=2)

                        if kernel_width != 0:
                            fhat['estimate'] = \
                                ndimage.gaussian_filter(fhat["estimate"],
                                                        sigma=kernel_width)

                        # Find a good threshold
                        fhat_star_peak_dens = fhat_stars["peaks_dens"]
                        clst_metadata["good_threshold"], _ = \
                            getDM.apply_peak_num_threshold(
                                fhat_star_peak_dens, fhat,
                                sig_fraction=sig_fraction)


                        # Apply threshold
                        threshold_mask = \
                            fhat["peaks_dens"] > clst_metadata['good_threshold']
                        fhat["peaks_dens"] = \
                            fhat["peaks_dens"][threshold_mask]
                        fhat["peaks_colIx"] = \
                            fhat["peaks_colIx"][threshold_mask]
                        fhat["peaks_rowIx"] = \
                            fhat["peaks_rowIx"][threshold_mask]
                        fhat["peaks_xcoords"] = \
                            fhat["peaks_xcoords"][threshold_mask]
                        fhat["peaks_ycoords"] = \
                            fhat["peaks_ycoords"][threshold_mask]

                        print ("putting peak info into h5")
                        peak_info_keys = \
                            ["peaks_xcoords", "peaks_ycoords", "peaks_rowIx",
                            "peaks_colIx", "peaks_dens"]

                        clst_metadata["good_threshold"] = \
                            '{0:0.10f}'.format(clst_metadata["good_threshold"])

                        peak_df = \
                            getgal.convert_dict_peaks_to_df(fhat,
                                                            clst_metadata,
                                                            peak_info_keys=peak_info_keys
                                                            )
                        store.append("peak_df", peak_df)

                        print ("putting fhat into h5")
                        getDM.convert_dict_dens_to_h5(fhat, clst_metadata,
                                                      h5_fstream)

print ("Done with all loops.")
h5_fstream.close()
store.close()
DM_fstream.close()
