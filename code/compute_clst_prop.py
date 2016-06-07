"""contains a bunch of functions for computing
properties from the subhalos of each cluster"""
from __future__ import division
import numpy as np


def subhalo_cent_dist(df):
    """calculate distance of subhalo from the most bound particle

    :param df: pandas dataframe, contains properties of one fof cluster
    """
    vec = np.array([df.SubhaloPos0,
                    df.SubhaloPos1,
                    df.SubhaloPos2])
    return [np.sqrt(np.dot(v, v)) for v in vec.transpose()]


def compute_relaxedness0(df, f, clstNo):
    """
    when fraction_of_Subf_subhalo_mass / M_FOF < 0.1
    the cluster is considered as relaxed.
    :param df: pandas dataframe, each row of the dataframe is extracted by
        extract_catalog.extract_clst(f, clstNo)
    :param f: hdf5 file stream of the main illustris fof data file
    :param clstNo: integer, denotes the cluster no with 0-based indexing
    """
    groupM = f["Group"]["GroupMass"][clstNo]
    subhaloM_total = np.sum(df["SubhaloMass"][1:])

    return subhaloM_total / groupM * 100.


def compute_non_relaxedness1(df, f, clstNo):
    """
    when relative_dist_{CM-most_bound} = dist(CM, most_bound) / R200C < 0.07
    the cluster is considered as relaxed.

    implementation notes:
        have to calculate the center of mass from subfind subhalos
        or we can use the weighted centroid as a substitute?

    :param df: pandas dataframe
        contains all the subhalo info of this cluster
    :param f: hdf5 file stream of the main hdf5 file
    :param clstNo: integer
    """
    center_of_mass = np.array([
        np.sum(df['SubhaloPos{i}'.format(i)] * df['SubhaloMass'])
                / np.sum(df['SubhaloMass'])
        for i in range (3)])

    dist_CM = np.sqrt(np.dot(center_of_mass, center_of_mass))
    R200C = f["Group/Group_R_Crit200"][clstNo]

    return dist_CM / R200C


# def compute_relaxedness1(df, f, clstNo):
#    """ relaxedness1 = total mass in subhalo / cluster mass * 100 < 10
#
#    :param df: pandas dataframe,
#        which is a subset of the subfind catalog that
#        contains properties of subhalo objects
#    :param f: hdf5 file stream
#    :param clstNo: integer
#
#    :returns relaxedness: float, percentage, relaxed if relaxedness < 10
#
#    :modified: df will be modified since dataframes are passed by reference
#        a key of "pos_dist" will be added
#        df["pos_dist"] : subhalo_dist / R200C
#
#    :notes: definition only applies for subhalos
#    """
#    groupM200C = f["Group"]["Group_M_Crit200"][clstNo]
#    R200C = f["Group"]["Group_R_Crit200"][clstNo]
#    df["pos_dist"] = subhalo_cent_dist(df) / R200C
#
# sum all the subhalos within R200C except the most bound subhalo
#    subhaloM200C = np.sum(df["SubhaloMass"][df["pos_dist"] < 1.][1:])
#
#    return subhaloM200C / groupM200C * 100.
