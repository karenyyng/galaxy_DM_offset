"""
For extracting data from raw halo FoF / subhalo catalog to catalog of clusters
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
            u'SubhaloLenType',
            u'SubhaloMass']


def extract_clst(f, clstNo, output=False, keys=default_keys(),
                 fix_phot_band=True,
                 outputFolder="../../data/", verbose=True):
    clst_df = pd.DataFrame(fix_clst_cat(f, clstNo, keys))

    # checks if positions are part of the keys
    for i in range(3):
        ckey = "SubhaloPos" + str(i)
        if ckey in clst_df.keys():
            # alternative syntax is to use
            # clst_df.apply(wrap_and_center_coord, blah blah)
            # potential speed up is to replace for loop with map operation
            clst_df.loc[:, ckey] = wrap_and_center_coord(clst_df[ckey])

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

    @stablity: to be tested
    """
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
                clst_dict[fix_key] = subhalos[key][i][ixes[0]: ixes[1]]

    return clst_dict


def wrap_and_center_coord(coords, edge_constraint=1e4, verbose=False):
    """ fixing the periodic boundary conditions per cluster
    wraps automatically at 75 Mpc / h then center coord at most bound particle

    :param coords: numpy array, denotes original coords
    :param verbose: bool

    :return: numpy array, coordinates that's been wrapped

    :stability: passed test
    """

    coords = np.array(coords)

    # need the - 7.4e4 part inside np.abs() since some might be closer
    # to the larger end of the box
    if np.all(np.abs(coords % 7.5e4 - 7.5e4) > edge_constraint):
        pass
    else:
        mask = coords > edge_constraint
        coords[mask] = coords[mask] - 7.5e4
        if verbose:
            print "close to box edge, needs wrapping"
            print "before masking ", coords[mask]
            print "after masking ", coords[mask]

    # needs to center coords - center on the most bound particle coords[0]
    # return coords - median(coords)
    return coords - coords[0]


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

    return None


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

    :return: None

    :stability: works

    :note: unclear to me that the numerical operations would be better
    if we stack dfs of different clusters before computing stat is better
    """
