"""Prototype script for final run for the DM.
Copied from `fhat_star_as_a_func_of_projection.py`

This
* reads the metadata for the cuts from the fhat_star*.h5 file
* produces the DM projections accordingly
"""
from __future__ import (print_function,
                        division, absolute_import)
import logging
import pandas as pd
import os
from scipy import ndimage
# from datetime import datetime
# datetime_stamp = datetime.now().strftime("%D").replace('/', '_')
datetime_stamp = '05_27_16'
data_path = "../../data/"

# ------- specify output file paths  -----------------------
total_clstNo = 2
input_datetime_stamp = datetime_stamp  # what fhat_star file to read in
logging_filename = "DM_logging_{0}_{1}.log".format(
    total_clstNo, datetime_stamp)
logging.basicConfig(filename=logging_filename, level=logging.INFO)
logging.info ("Current date is {}".format(datetime_stamp))

# ----------------------------------------------------------
output_fhat_filename = \
    "test_DM_fhat_clst{0}_{1}.h5".format(total_clstNo, datetime_stamp)

logging.info ("Outputing files to: " + output_fhat_filename)

import numpy as np
import sys
from collections import OrderedDict
sys.path.append("../")

import extract_catalog as ec
import get_DM_centroids as getDM
import get_gal_centroids as getgal
import get_KDE as getKDE

verbose = True
import h5py
DM_h5file = data_path + \
    "Illustris-1_00135_APillepich_KarenNG_ParticleData_Group_PartType1.h5"

DM_fstream = h5py.File(DM_h5file)
# ================ make all decisions ===========================
# Specify fhat_star input file paths
# input_star_file = \
#     "test_stars_peak_df_clst{}_{}.h5".format(
#         total_clstNo, input_datetime_stamp)
# input_h5_key = "peak_df"
# DM_metadata, star_peak_df = \
#     getDM.retrieve_DM_metadata_from_gal_metadata(
#         data_path, input_star_file, input_h5_key)

input_star_fhat_file = data_path + \
    "test_stars_fhat_clst{}_{}.h5".format(
        total_clstNo, input_datetime_stamp)
fhats_stars = h5py.File(input_star_fhat_file, "r")
logging.info ("input_star_file = " + input_star_fhat_file)


DM_metadata = \
    getDM.retrieve_DM_metadata_from_gal_h5file(fhats_stars)
DM_metadata["projection"] = \
    np.unique([str(p) for p in DM_metadata["projection"]])

# need the DM_metadata to be sorted
DM_metadata['clstNo'] = sorted(DM_metadata['clstNo'])

logging.info("ClstNo to be analyzed from reading metadata")
logging.info("DM_metadata['clstNo'] = {}".format(DM_metadata['clstNo']))

# star_gpBy, star_gpBy_keys = \
#     getgal.get_clst_gpBy_from_DM_metadata(star_peak_df)

DM_resolution = 2.  # kpc
DM_metadata["kernel_width"] = np.array([0, 50]) / DM_resolution
DM_metadata["kernel_width"] = ['{0:0.1f}'.format(stuff) for stuff in
                               DM_metadata["kernel_width"] ]
DM_metadata["sig_fraction"] = [0.2]

# ============== set up output file structure  ===========

# Check_metadata against illegal types
# Create HDF5 file structure first!
if os.path.isfile(data_path + output_fhat_filename):
    os.remove(data_path + output_fhat_filename)

logging.info ("{} projections per cluster are constructed.".format(
    len(DM_metadata["projection"])))

h5_fstream = \
    getDM.construct_h5_file_for_saving_fhat(
        DM_metadata, output_fhat_filename, output_path=data_path
        )

# ============== prepare data based on the metadata ===========
pos_cols = ["SubhaloPos{}".format(i) for i in range(3)]

clst_metadata = OrderedDict({})

# #### CHANGE THE RANGE of line below
for i, clstNo in enumerate(DM_metadata["clstNo"]):
    logging.info ("Processing {0}-th ".format(i) +
           "out of the range {0} to {1}".format(DM_metadata['clstNo'][0],
                                                DM_metadata['clstNo'][-1]
                                                ))
    peak_df = pd.DataFrame()
    star_path = ''
    clst_metadata["clstNo"] = clstNo  # clstNo is a string
    clstNo = int(clstNo)

    # Get coordinates without shifting coordinates to be positive
    coord_dict = \
        ec.get_DM_particles([clstNo], DM_fstream, dataPath=data_path,
                            shift_coords_for_hist=False
                            )[clstNo]

    # There are no cuts / weights for the DM particles.
    # However, the DM peaks are informed by the galaxy peaks.
    # The choice of cuts and weights affect the galaxy peaks.
    for cut in DM_metadata["cut"]:
        clst_metadata['cut'] = cut

        for weights in DM_metadata["weights"]:
            clst_metadata['weights'] = weights

            # los = line of sight
            for los_axis in DM_metadata["los_axis"]:
                clst_metadata["los_axis"] = los_axis
                # We save the meta data as string but needs int for los_axis
                los_axis = int(los_axis)

                # make sure there is no duplicate projection due to duplicate
                # kernel width
                for i in range(len(DM_metadata["projection"])):
                    clst_metadata["projection"] = \
                        eval(DM_metadata["projection"][i])
                    star_path = str(clstNo) + '/' + cut + '/' + weights + \
                        '/' + str(los_axis) + '/' + \
                        str(clst_metadata["projection"]) + "/"

                    data = getgal.project_coords(
                        coord_dict["coords"],
                        clst_metadata["projection"][0],
                        clst_metadata["projection"][1],
                        los_axis=clst_metadata["los_axis"])

                    # Retain the 2 spatial axes that is not the los_axis
                    col = np.arange(data.shape[1]) != los_axis
                    data = data[:, col]

                    # Put data in a dictionary to pretend to have
                    # the same layout as required by the function
                    # `make_histogram_with_some_resolution`
                    data = {"coords": data}
                    data['min_coords'] = np.min(data["coords"], axis=0)
                    data['coords'] = data['coords'] - data['min_coords']

                    fhat = \
                        getDM.make_histogram_with_some_resolution(
                            data, resolution=2,  # in kpc
                            find_peak=False)

                    #### CHANGE THE RANGE of DM_metadata["kernel_width"] below
                    for kernel_width in DM_metadata["kernel_width"]:

                        # Find the correpsonding galaxy density projection
                        fhat_stars = fhats_stars[star_path]

                        clst_metadata["kernel_width"] = kernel_width
                        logging.info (
                            "kernel_width = {}".format(kernel_width))
                        if float(kernel_width) > 0.:
                            fhat['estimate'] = \
                                ndimage.gaussian_filter(
                                    fhat["estimate"],
                                    sigma=float(kernel_width))

                        getKDE.find_peaks_from_py_diff(fhat)
                        getKDE.get_density_weights(fhat)

                        # Fixed threshold for calling a gal peak as significant
                        accepted_peak_no = getDM.apply_peak_num_threshold(
                            fhat_stars["peaks_dens"][:],
                            fhat,
                            sig_fraction=DM_metadata["sig_fraction"][0]
                        )

                        # only entries listed in peak_info_keys get stored
                        peak_info_keys = \
                            ["peaks_xcoords",
                             "peaks_ycoords",
                             # "peaks_rowIx",
                             # "peaks_colIx",
                             "peaks_dens"]

                        # Apply the threshold to reduce the amount of peaks
                        # stored
                        for peak_property in peak_info_keys:
                            fhat[peak_property] = \
                                fhat[peak_property][:accepted_peak_no]

                        fixed_size_data_keys = [
                            "eval_points",
                            "estimate",
                            "peaks_xcoords",
                            "peaks_ycoords",
                            "peaks_dens"
                        ]

                        logging.info ("Putting fhat into h5")
                        getDM.convert_dict_dens_to_h5(
                            fhat, clst_metadata, h5_fstream,
                            verbose=False,
                            fixed_size_data_keys=fixed_size_data_keys
                        )

logging.info ("Done with all loops.")
h5_fstream.close()
DM_fstream.close()
