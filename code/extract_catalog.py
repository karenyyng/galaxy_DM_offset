"""Extract data from raw halo FoF / subhalo catalog to catalog of clusters.
for the Illustris-1 simulation
Author: Karen Ng
License: BSD
"""

import sys
sys.path.append("../IEnv/lib/python2.7/site-packages/")  # virtualenv path
import numpy as np
import h5py
import pandas as pd


def default_keys():
    """ select some keys to be extracted from h5 files to df """
    return [u'SubhaloPos',
            # u'SubhaloCM',
            # u'SubhaloHalfmassRad',
            # u'SubhaloHalfmassRadType',
            # u'SubhaloParent',
            u'SubhaloGrNr',
            u'SubhaloStellarPhotometrics',
            # u'SubhaloStellarPhotometricsMassInRad',
            u'SubhaloLenType',
            u'SubhaloMass',
            # u'SubhaloMassType'
            ]


def extract_clst(f, clstNo, output=False, keys=default_keys(),
                 fix_phot_band=True, outputFolder="../../data/", verbose=True):

    clst_df = pd.DataFrame(fix_clst_cat(f, clstNo, keys))

    # checks if positions are part of the keys
    for i in range(3):
        ckey = "SubhaloPos" + str(i)
        if ckey in clst_df.keys():
            # alternative syntax is to use
            # clst_df.apply(wrap_and_center_coord, blah blah)
            # potential speed up is to replace for loop with map operation
            clst_df.loc[:, ckey] = \
                wrap_and_center_coord(clst_df[ckey])   # , clst_df[ckey][0])

    fix_phot_band_names(clst_df)

    if output:
        outputFile = outputFolder + "/cluster_" + str(clstNo) + ".h5"
        if verbose:
            print "outputting file :" + outputFile
            print "with key name as df"
        clst_df.to_hdf(outputFile, "df")

    return clst_df


def get_subhalos_of_a_cluster(f, clstID):
    """
    :params f = file connection to HDF5 file
    :params clstID = integer, denotes the cluster ID as 0, 1, 2, etc.

    :returns a list of two integers,
        first integer denotes the index of the first subhalo of the
        cluster, second denotes the index of the last subhalo of the
        cluster

    @stablity: ok!
    """
    clstID = int(clstID)
    return [int(f["Group"]["GroupFirstSub"][ID])
            for ID in [clstID, clstID + 1]]


def fix_phot_band_names(df):
    """rename key names to be more informative"""
    photo = "SubhaloStellarPhotometrics"
    bands = ["U", "B", "V", "K", "g", "r", "i", "z"]
    bands = [b + "_band" for b in bands]
    phot_bands = {photo + str(i): bands[i] for i in range(len(bands))}

    for replacement in phot_bands.iteritems():
        if replacement[0] in df.keys():
            df.rename(columns={replacement[0]: replacement[1]}, inplace=True)

    return None


def fix_clst_cat(f, clstNo, keys=default_keys()):
    """ fix the weird shapes in the f["Subhalo"] files
    :param
    subhalos = hdf5 file stream, e.g. f["Subhalo"]
    clstNo = integer, denotes the parent halo ID as 0, 1, 2, etc.
    keys = list of strings denoting relevant keys

    :returns
    dictionary with relevant keys,
    if the number of features of the key is just one,
    the returned key is the same as the original key
    if not, the key is original key concatenated with the number of the
    feature

    :stability works
    """
    subhalos = f["Subhalo"]
    clst_dict = {}
    ixes = get_subhalos_of_a_cluster(f, clstNo)

    for key in keys:
        # if that key only has 1 dimension then just put it in df with
        # original key name
        if np.ndim(subhalos[key]) == 1:
            clst_dict[key] = subhalos[key][ixes[0]: ixes[1]]
        else:  # create more sub-keys for that particular key
            for i in range(subhalos[key].shape[0]):
                fix_key = key + str(i)
                clst_dict[fix_key] = subhalos[key][i, ixes[0]: ixes[1]]
                #clst_dict[fix_key] = subhalos[key][i][ixes[0]: ixes[1]]

    return clst_dict


def wrap_and_center_coord(coords, edge_constraint=1e4,
                          verbose=False, center_coord=None,
                          wrap_criteria=7.5e4):
    """ fixing the periodic boundary conditions per cluster
    wraps automatically at 75 Mpc / h for stars then center coord at most bound particle
    The cluster should not be 10 Mpc within the edge

    :note: this meant to be applied to each spatial dimension

    :param coords: numpy array, denotes original coords
    :param verbose: bool

    :return: numpy array, coordinates that's been wrapped

    :stability: passed test

    Units = kpc / h
    """

    coords = np.array(coords)

    # need the - wrap_criteria part inside np.abs() since some might be closer
    # to the larger end of the box
    #if np.all(np.abs(coords % wrap_criteria - wrap_criteria) > edge_constraint):
    not_close_to_edge = \
        np.abs(coords % wrap_criteria - wrap_criteria) > edge_constraint
    if np.all(not_close_to_edge):
        pass
    else:
        mask = coords > edge_constraint
        if verbose:
            print "Close to box edge, needs wrapping"
            print "before masking ", coords[mask]
        coords[mask] = coords[mask] - wrap_criteria
        if verbose:
            print "after masking ", coords[mask]

    # needs to center coords - center on the most bound particle coords[0]
    if center_coord is None:
        return coords - coords[0]
    else:
        assert center_coord == float, \
            "wrong dimension - this is supposed to be a float"
        return coords - center_coord


def add_info(h5, info, h5_key="df", h5_subkey="info"):
    """
    :param
    h5 = string, full file path with extension
    info = string, info about file that you wish to add to the dataframe

    :return None
    note: by default the key to that you add to is "df"
    """
    h5_df = h5py.File(h5, "a")
    h5_df[h5_key].attrs[h5_subkey] = info
    h5_df.close()
    print "closing file"

    return None


# ----------helper functions for getting DM particle data --------------------
def get_DM_particles(clsts, partDMh5, h5part_coord_key="PartType1_Coordinates",
                     partDMhaloIxLoc_h5file="DM_particles_clst_loc.h5",
                     dataPath="../data/",verbose=False,
                     shift_coords_for_hist=True):
    """This reads particle data from the h5 file stream, wraps and
    the coordinates of the particles.

    Parameters
    ----------
    clsts : list of integer(s), first clst has an ID of 0
        indicates which cluster(s) you would like to retrieve
    partDMh5 : hdf5 files stream
        this connects to the hdf5 file that contains all DM particle data
    h5part_coord_key : str
        this is the key that contains coordinate information
    partDMhaloIxLoc_h5file : str
        file name of the hdf5 file that contains the location of index of the
        last DM particle of each halo
    dataPath : str (optional)
        where the data files are stored and is appended before file names
    shift_coords_for_hist: bool (optional), default = True
        whether if the spatial coordinates of histogram bins should be shifted
        to be > 0

    Returns
    ------
    dictionary
        key : clst(s) id
        value : numpy array
            with the coordinates of the particles of that
            particular clst.
        * Coordinates are first centered,
        * then the min. coordinates are subtracted to make sure all coordinates are
        positive so we can make histograms with matplotlib
    """
    # Make sure we do not write to the file by using read-only mode
    part_halos = h5py.File(dataPath + partDMhaloIxLoc_h5file, "r")

    # Add 0 as first element of list so we can specify the range of location
    # with the syntax below correctly
    # We retrieve the end indices of all the particles within a cluster
    haloEndIx = [0] + list(part_halos["loc"][...])
    # Extract coordinates from relevant parts of the h5 file
    coords = {clstNo: partDMh5[h5part_coord_key][:, haloEndIx[clstNo]:
                                                 haloEndIx[clstNo + 1]]
              for clstNo in clsts}

    # Wrap and center coordinates then put it in a dictionary
    coords = {clstNo: {"coords":
                       np.array(map(lambda x:
                                    wrap_and_center_coord(x,
                                                          edge_constraint=1e4,
                                                          wrap_criteria=
                                                          7.5e4 / 0.704,
                                                          verbose=verbose), crd,
                                    ))}
              for clstNo, crd in coords.iteritems()}

    # Compute min. coords and make (0, 0) to be min. coords for 2D histogram
    # Put the min. coords in the correct dictionary
    # so we can shift all coordinates to original frame for the correct cluster
    # later on
    for clstNo, cl_dict in coords.iteritems():
        if shift_coords_for_hist:
            cl_dict["min_coords"] = \
                np.min(cl_dict["coords"], axis=1)\
                .reshape(cl_dict["coords"].shape[0], 1)

            cl_dict["coords"] = (cl_dict["coords"] -
                                cl_dict["min_coords"]).transpose()
        else:
            cl_dict["coords"] = cl_dict["coords"].transpose()


    return coords


def check_particle_parent_clst(particle_halo_id, clsts=129,
                               start_id=0, end_id=8e7):
    """
    :param particle_halo_id: hdf5 fstream array
        contains particle parent halo id
    :param clsts: integer
        what clusters you would like to get the particle ID location
    :param start_id: integer
        which index of the `particle_halo_id` array  you would like to limit
        the search range to
    :param start_id: integer
        which index of the `particle_halo_id` array you would like to limit
        the search range to
    """
    loc = []
    count = 0
    clst_list = range(1, clsts)
    clst_to_match = clst_list.pop(0)
    for i in particle_halo_id[start_id:end_id]:
        if int(i) != clst_to_match:
            # print("i = ", i)
            # print("clst_to_match=", clst_to_match)
            loc.append(count)
            if clst_list:
                clst_to_match = clst_list.pop(0)
            else:
                return loc
        count += 1
    return loc



# -------------docstrings --------------------------------------
extract_clst.__doc__ = \
    """calls function to extract clst as dataframe
    this
    * fixes weird shapes in each key
    * wraps clusters at the end of the periodic box
    * centers clusters
    * fixes the names of photometric bands to be more informative
    * can choose to output df to hdf5 files

    :param keys: list of strings, denoting relevant keys
    :param f: file stream object, connected to a HDF5 file, usage: f["Subhalo"]
    :param clstNo: integer, denotes the parent halo ID ordered by mass,
        e.g. 0, 1, 2, 3, ...
        use map for extracting a list of clusters
    :param output: bool, whether to save data to hdf5 file or not
    :param outputFolder: string, denotes output directory
    :fix_phot_band: bool, whether to change the names of photometric bands
    :verbose: bool, if printing is wanted

    :return: clst_df

    :stability: works

    :example:
    >>> # this extracts the most massive clusters with mass > 1e13 solar mass
    >>> map(lambda i: extract_clst(f, i), range(129))

    :note: unclear to me that the numerical operations would be better
    if we stack dfs of different clusters before computing stat is better
    """
