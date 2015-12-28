"""
This is a script for plotting the cummulative distribution of mass
for 129 `clusters`.
"""


# def plot_cumulative_mass_distribution(groupMass, groupMcrit200, groupMcrit500,
#                                       ax=None):
#     from plot_clst_prop import *
#     ticks, countGroupMass = compute_clst_no_above_mass_threshold(groupMass)
#     ticks, countGroupMcrit200 = compute_clst_no_above_mass_threshold(groupMcrit200)
#     ticks, countGroupMcrit500 = compute_clst_no_above_mass_threshold(groupMcrit500)
#
#     y_data = [countGroupMass, countGroupMcrit200, countGroupMcrit500]
#     y_legend = [r"$M_{\rm FoF}$", r"$M_{200c}$", r"$M_{500c}$"]
#     x_label = r"$M_{Cluster}(M_{\odot})$"
#     y_label = r"$N(> M_{Cluster})$"
#
#     plot_cluster_mass_distribution(ticks, y_data,
#                                    y_legend, x_label, y_label, ax=ax,
#                                    save=True, path="../../paper/figures/drafts/")
#
#     if close:
#         f.close()


def plot_mass_richness_relationship(mass, richness):
    """
    :mass: numpy array of floats, (FoF) masses of clusters
    :richness: numpy array of int, richness after a certain cut
    :returns:
    """

    return


if __name__ == "__main__":
    import h5py
    import sys
    sys.path.append("../")

    close = True
    h5File = \
        "../../data/" + \
        "Illustris-1_fof_subhalo_myCompleteHaloCatalog_00135.hdf5"

    f = h5py.File(h5File, "r")

    groupMass = f["Group"]["GroupMass"][:]
    groupMcrit200 = f["Group"]["Group_M_Crit200"][:]
    groupMcrit500 = f["Group"]["Group_M_Crit500"][:]
    plot_cumulative_mass_distribution(groupMass, groupMcrit200, groupMcrit500)
