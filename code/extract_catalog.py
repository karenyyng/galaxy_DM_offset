"""
For extracting from raw halo FoF / subhalo catalog to catalog of clusters
for the Illustris-1 simulation
Author: Karen Ng
License: BSD
"""
import sys
sys.path.append("/Users/karenyng/Documents/Illustris_analysis/IEnv/lib/" +
                "python2.7/site-packages")

import numpy as np
#import pandas as pd
#import h5py


def extract_clst(keys, subhalos, clstNo, clstMemMask, output=False,
                 outputFolder="../../data/", verbose=True):
    """calls function to fix weird shape
    can choose to output df to hdf5 files

    parameters:
    ===========
    keys = list of strings denoting relevant keys
    subhalos = hdf5 file stream, e.g. f["Subhalo"]
    clstNo = integer, denotes the parent halo ID ordered by mass,
        e.g. 0, 1, 2, 3, ...
    clstMemMask = dictionary of np array of bools,
        denotes which subhalo belongs to the parent halo ID
        key is the parent subhalo ID
        value is the list of bools that tells you which subfind objects
        belong to that parent subhalo
    outputFolder = string, denotes output directory
    verbose = bool, if printing is wanted

    returns:
    ========
    None

    Stability: works
    """
    clst_dict = \
        fix_clst_cat(keys, subhalos, clstNo, clstMemMask)
    clst_df = pd.DataFrame(clst_dict)

    if output:
        outputFile = outputFolder + "cluster_" + str(clstNo) + ".h5"
        if verbose:
            print "outputting file :" + outputFile
        clst_df.to_hdf(outputFile, "df")

    return clst_df


def fix_clst_cat(relevantSubhaloKeys, subhalos, clstNo, clstMemMask):
    """ fix the weird shapes in the f["Subhalo"] files
    parameters:
    ===========
    relevantSubhaloKeys = list of strings denoting relevant keys
    subhalos = hdf5 file stream, e.g. f["Subhalo"]
    clstNo = integer, denotes the parent halo ID
    clstMemMask = dictionary of np array of bools,
        denotes which subhalo belongs to the parent halo ID
        key is the parent subhalo ID
        value is the list of bools that tells you which subfind objects
        belong to that parent subhalo

    returns:
    ========
    dictionary with relevant keys,
    if the number of features of the key is just one,
    the returned key is the same as the original key
    if not, the key is original key concatenated with the number of the
    feature

    Stability: works
    """
    clst_dict = {}
    for key in relevantSubhaloKeys:
        if subhalos[key].shape[0] == clstMemMask[clstNo].size:
            clst_dict[key] = \
                subhalos[key][clstMemMask[clstNo]][...]
        else:
            for i in range(subhalos[key].shape[0]):
                fix_key = key + str(i)
                clst_dict[fix_key] = \
                    subhalos[key][i][...][clstMemMask[clstNo]]

    return clst_dict


def wrap_and_center_coord(coords, edge_constraint=1e4,
                          verbose=False):
    """ fixing the periodic boundary conditions per cluster
    wraps automatically at 75 Mpc / h then center coord at cluster center
    @param coords = numpy array, denotes original coords
    @param verbose = bool

    @return numpy array, coordinates that's been wrapped

    @stability : to be tested
    """

    coords = np.array(coords)

    if np.all(np.abs(coords % 7.5e4 - 7.5e4) > edge_constraint):
        pass
    else:
        mask = coords > edge_constraint
        coords[mask] = coords[mask] - 7.5e4
        if verbose:
            print "close to box edge, needs wrapping"
            print "before masking " , coords[mask]
            print "after masking " , coords[mask]

    # needs to center coords - use median to be the "center"
    coords = coords - np.median(coords)

    return coords


def add_info(h5, info):
    """
    h5 = file
    info = string, info about file
    """
    h5_df = h5py.File(h5, "a")
    h5_df["df"].attrs["info"] = info
    #"most massive cluster extracted from " + \
    #"Illustris-1_fof_subhalo_myCompleteHaloCatalog_00135.hdf5"
    h5_df.close()

    return None


