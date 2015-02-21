import h5py
import numpy as np
import sys
import time

sys.path.append("../")

import extract_catalog as ec
# following line only works when you are in the ks_KDE.r directory
import get_gal_centroids as cent
import compute_clst_prop as cp
import pickle
import sys

h5File = "../../data/Illustris-1_fof_subhalo_myCompleteHaloCatalog_00135.hdf5"
f = h5py.File(h5File, "r")

# there are only 129 clst > 1e13 Msun
allClst = int(sys.argv[1])
suffix = str(sys.argv[2])

print "examining {0} clusters in total".format(allClst)

df_list = [ec.extract_clst(f, clstNo) for clstNo in range(allClst)]

cut_kwargs = {'DM_cut': 1e3, 'star_cut': 5e1}

result_list = [cent.compute_KDE_peak_offsets(df_list[i], f, i,
                                             cent.cut_reliable_galaxies,
                                             cut_kwargs,
                                             w=df_list[i]["SubhaloMass"])
               for i in range(allClst)]

offsets_list = [result_list[i][:2] for i in range(allClst)]
fhat_list = [result_list[i][2] for i in range(allClst)]

relaxedness_list = [cp.compute_relaxedness0(df_list[i], f, i) for i in
                    range(allClst)]

f = open('offset_list_{0}'.format(clstNo) + suffix + '.pkl'.format(allClst),
         'w')
pickle.dump(offsets_list, f)
f.close()

f = open('relaxedness_{0}'.format(clstNo) + suffix + '.pkl'.format(allClst),
         'w')
pickle.dump(relaxedness_list, f)
f.close()

f = open('fhat_{0}'.format(clstNo) + suffix + '.pkl'.format(allClst), 'w')
pickle.dump(fhat_list, f)
f.close()
