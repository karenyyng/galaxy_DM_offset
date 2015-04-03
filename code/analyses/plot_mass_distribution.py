import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import sys
sys.path.append("../")
from plot_clst_prop import *

h5File = \
    "../../data/" + \
    "Illustris-1_fof_subhalo_myCompleteHaloCatalog_00135.hdf5"

f = h5py.File(h5File, "r")

groupMass = f["Group"]["GroupMass"][...]
groupMcrit200 = f["Group"]["Group_M_Crit200"][...]
groupMcrit500 = f["Group"]["Group_M_Crit500"][...]

ticks, countGroupMass = compute_clst_no_above_mass_threshold(groupMass)
ticks, countGroupMcrit200 = compute_clst_no_above_mass_threshold(groupMcrit200)
ticks, countGroupMcrit500 = compute_clst_no_above_mass_threshold(groupMcrit500)

y_data = [countGroupMass, countGroupMcrit200, countGroupMcrit500]
y_legend = [r"$M_{\rm FoF}$", r"$M_{200c}$", r"$M_{500c}$"]
x_ticks = r"$M_{Cluster}(10^{10} M_{\odot})$"
y_ticks = r"$N(> M_{Cluster})$"

plot_cluster_mass_distribution(ticks, y_data,
                               y_legend, x_ticks, y_ticks,
                               save=True, path="../../paper/figures/drafts/")
