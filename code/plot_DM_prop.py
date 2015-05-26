import matplotlib.pyplot as plt
import numpy as np


def plot_DM_fhat(fhat, clstNo, peaks_mask):
    plt.figure(figsize=(20, 20))
    plt.axes().set_aspect('equal')
    plt.title("Clst {}".format(clstNo), size=20)
    plt.xlabel("kpc", size=30)
    plt.ylabel("kpc", size=30)
    plt.tick_params(labelsize='large')

    plt.contourf(fhat["eval_points"][0], fhat["eval_points"][1],
                 fhat["estimate"].transpose(), cmap=plt.cm.afmhot)

    ixes = np.arange(0, len(fhat["peaks_xcoords"]), 1)[peaks_mask]
    plt.plot(fhat["peaks_xcoords"][ixes], fhat["peaks_ycoords"][ixes],
             "ro", fillstyle="none", mew=1.5, ms=20,
             label="identified peaks")

    plt.legend(loc='best', frameon=True, fancybox=True)
    return
