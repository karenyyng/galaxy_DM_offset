
from __future__ import (print_function, unicode_literals,
                        division, absolute_import)

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
sys.path.append("../")

import extract_catalog as ext_cat
# import get_gal_centroids as getg
# import plot_clst_prop as plotClst
from plot_clst_prop import visualize_3D_clst
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser(
        description="Visualize DM subhalos in 3D for specific cluster.")
parser.add_argument("clstNo", metavar="cluster_no", type=int,
                    help="Zeroth based numbering of cluster number")
args = parser.parse_args()

dataPath = "../../data/"
original_f = h5py.File(dataPath +
                "Illustris-1_fof_subhalo_myCompleteHaloCatalog_00135.hdf5")

fig = plt.figure()
ax = Axes3D(fig)
df = ext_cat.extract_clst(original_f, args.clstNo)
position_keys = [k for k in df.keys() if "SubhaloPos" in k]

ax.plot(df[position_keys[0]], df[position_keys[1]], df[position_keys[2]],
        'o', alpha=0.05, label='subhalo location')

ax.legend(loc='best')
plt.show()



# final_los_along_x = df[pos][:10000].apply(lambda x:
#          getg.project_coords(x, xi=0, phi=0,
#                              los_axis=4), axis=1)
# final_los_along_x = np.array(final_los_along_x)
# plt.axes().set_aspect('equal')
# plt.plot(final_los_along_x[:, 0],
#          final_los_along_x[:, 1], '.', alpha=0.5)
# plt.xlabel('y')
# plt.ylabel('z')
