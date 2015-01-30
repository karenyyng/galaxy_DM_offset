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
    groupM = f["Group"]["GroupMass"][clstNo]
    subhaloM_total = np.sum(df["SubhaloMass"][1:])

    return subhaloM_total / groupM * 100.


def compute_non_relaxedness1():
    return


#def compute_relaxedness1(df, f, clstNo):
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
#    # sum all the subhalos within R200C except the most bound subhalo
#    subhaloM200C = np.sum(df["SubhaloMass"][df["pos_dist"] < 1.][1:])
#
#    return subhaloM200C / groupM200C * 100.
