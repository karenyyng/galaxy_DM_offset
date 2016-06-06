# import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from statsmodels.api import nonparametric
from astropy.stats import biweight_location as C_BI
from types import ModuleType


def kde1d(x, ax, prob=None, kernel="gau", bw="scott", fft=True,
          gridsize=None, weights=None,
          adjust=1, cut=3, clip=(-np.inf, np.inf), xlabel=None, lim=None,
          labelsize=None, legendloc=None, **kwargs):
    """wrapper around the statsmodel.api.nonparametric.KDEUnivariate()
    and plot

    parameters
    ==========
    ax = matplotlib axis object, e.g. ax created by plt.subplots() or
        matplotlib.pyplot
    x = numpy array of numerical values, the data that you try to visualize
    prob = numpy array, has the same length as x
    kernel = string, what kernel to use, see KDEUnivariate() documentation
        for options
    bw = string or integer, denotes the binwidth
    fft = bool, if fast fourier transform should be used while
        computing the kde
    see KDEUnivaraiate documentation for the rest of the parameters

    label = string, denotes what would be put as xlabel of the plot
    lim = tuple of length 2 with floats, denotes the xlim on the plot

    returns
    =======
    support = numpy array
        the corresponding x value of each element of the returned pdf
    pdf = numpy array, the corresponding probability distribution function
        same length as support
    """
    rc("font", family="serif")
    kde = nonparametric.KDEUnivariate(x)
    kde.fit(kernel, bw, fft, gridsize=gridsize, cut=cut, clip=clip,
            weights=weights)
    support, density = kde.support, kde.density

    ax.plot(support, density, 'b-', **kwargs)
    if isinstance(ax, ModuleType):
        ax.ylabel("PDF")
        if xlabel is not None:
            ax.xlabel(xlabel, fontsize=labelsize)
        if lim is not None:
            ax.xlim(lim)
        if legendloc is not None:
            ax.legend(loc=legendloc)
    else:
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if lim is not None:
            ax.set_xlim(lim)

    return support, density


def central_CI(support, density, level=68, lim=None):
    """returns the central credible intervals for a kde estimate from kde1d
    of a posterior
    parameters
    ==========
    support = numpy array
    density = numpy array that has the same length as the support
    level = float, indicates what percentile to include between
    lim = tuple of float of length 2

    returns
    ======
    low_ix = index of the lower limit, support[low_ix] gives the estimate
    up_ix = index of the upper limit, support[up_ix] gives the estimate

    warning: the index may be off by one depending on how you plot
    but kde is smooth enough it should not matter

    stability:
    ========
    work in progress
    """
    # TODO should checks for a sane limit

    if lim is not None:
        lix = 0
        while(support[lix] < lim[0]):
            lix += 1
    else:
        lix = 0

    sig = (1 - level / 100.) / 2.
    total = np.sum(density[lix:])
    exclude_reg = total * sig

    low_exc = 0
    low_ix = 0
    while(low_exc < exclude_reg):
        low_exc += density[low_ix]
        low_ix+=1

    up_exc = 0
    up_ix = density.size - 1
    while(up_exc < exclude_reg):
        up_exc += density[up_ix]
        up_ix-=1
    return low_ix, up_ix


def CI_loc_plot(x, ax, c='b', prob=None, kernel="gau", bw="silverman",
                fft=True, gridsize=None, adjust=1, cut=3,
                clip=(-np.inf, np.inf), xlabel=None, lim=None,
                weights=None, ylabel=None,
                labelsize=None, legendloc=None, **kwargs):

    # make font looks like latex font
    rc("font", family="serif")
    support, den = kde1d(x, ax, prob=prob, kernel=kernel, bw=bw,
                         fft=fft, gridsize=gridsize, adjust=adjust, cut=cut,
                         clip=clip, xlabel=xlabel, lim=lim,
                         weights=weights,
                         labelsize=labelsize, legendloc=legendloc,
                         **kwargs)
    low68_ix, up68_ix = central_CI(support, den, level=68, lim=lim)
    low95_ix, up95_ix = central_CI(support, den, level=95, lim=lim)
    ax.fill_between(support[low68_ix: up68_ix],
                    den[low68_ix: up68_ix], alpha=0.5, color=c)
    ax.fill_between(support[low95_ix: up95_ix],
                    den[low95_ix: up95_ix], alpha=0.2, color=c)
    loc_ix = low68_ix
    loc = C_BI(x)
    while(support[loc_ix] < loc):
        loc_ix += 1
    ylim = ax.get_ylim()
    # xlim = ax.get_xlim()
    ax.axvline(loc, ymin=0.0,
               ymax=(den[loc_ix] + den[loc_ix + 1]) / 2. / ylim[1],
               ls='--', lw=3, c='k')

    ax.set_ylim(0., ylim[1])

    sum_stat = {"loc": loc,
                "low68": support[low68_ix],
                "up68": support[up68_ix],
                "low95": support[low95_ix],
                "up95": support[up95_ix],
                }

    return sum_stat


def N_by_N_lower_triangle_plot(data, space, var_list, axlims=None,
                               Nbins_2D=None, axlabels=None, N_bins=None,
                               xlabel_to_rot=None, histran=None, figsize=6,
                               fontsize=12, save=False, prefix=None,
                               suffix=".png", path="./"):
    """ create a N by N matrix of plots
    with the top plot of each row showing a density plot in 1D
    and the remaining plots being 2D contour plots
    df = dataframe that contain the data of all the variables to be plots
    space = float, px of space that is added between subplots
    var_list = list of strings - denotes the column header names
        that needs to be plotted
    axlims = dictionary, keys are the strings in var_list,
        each value is a tuple of (low_lim, up_lim) to denote the limit
        of values to be plotted
    Nbins_2D = dictionary, keys are in format of tuples of
        (x_col_str, y_col_str) to denote which subplot you are referring to
    axlabels = dictionary, keys correspond to the variable names
    xlabel_to_rot = dictionary,
        key is the the key for the labels to be rotated,
        value is the degree to be rotated
    histran = dictionary,
        some keys has to be the ones for the plots, value are in
        form of (lowerhist_range, upperhist_range)
    figsize = integer, figuares are squared this refers to the side length
    fontsize = integer, denotes font size of the labels
    save = logical, denotes if plot should be saved or not
    prefix = string, prefix of the output plot file
    path = string, path of the output plot file
    suffix = string, file extension of the output plot file

    Stability: Not entirely tested, use at own risk
    """
    from matplotlib.ticker import MaxNLocator

    def comb_zip(ls1, ls2):
        return [(lb1, lb2) for lb1 in ls1 for lb2 in ls2]

    # begin checking if inputs make sense
    N = len(var_list)
    assert N <= len(axlabels), "length of axlabels is wrong"
    assert N >= 2, "lower triangular contour plots require more than 2\
        variables in the data"

    for var in var_list:
        assert var in data.columns, "variable to be plotted not in df"

    if axlabels is None:
        axlabels = {key: key for key in var_list}

    if xlabel_to_rot is None:
        xlabel_to_rot = {key: 0 for key in var_list}

    if histran is None:
        histran = {key: None for key in var_list}

    if axlims is None:
        axlims = {key: (None, None) for key in var_list}

    if Nbins_2D is None:
        keys = comb_zip(var_list, var_list)
        Nbins_2D = {key: 50 for key in keys}

    if N_bins is None:
        N_bins = {key: 'knuth' for key in var_list}

    if save:
        assert prefix is not None, "prefix for output file cannot be none"

    # impossible for the matrix plot not to be squared in terms of dimensions
    # set each of the subplot to be squared with the figsize option
    f, axarr = plt.subplots(N, N, figsize=(figsize, figsize))
    f.subplots_adjust(wspace=space, hspace=space)

    # remove unwanted plots on the upper right
    plt.setp([a.get_axes() for i in range(N - 1)
              for a in axarr[i, i + 1:]], visible=False)

    # remove unwanted row axes tick labels
    plt.setp([a.get_xticklabels() for i in range(N - 1)
              for a in axarr[i, :]], visible=False)

    # remove unwanted column axes tick labels
    plt.setp([axarr[0, 0].get_yticklabels()], visible=False)
    plt.setp([a.get_yticklabels() for i in range(N - 1)
              for a in axarr[i + 1, 1:]], visible=False)

    # create axes labels
    if axlabels is not None:
        for j in range(1, N):
            axarr[j, 0].set_ylabel(axlabels[var_list[j]], fontsize=fontsize)
        for i in range(N):
            axarr[N - 1, i].set_xlabel(axlabels[var_list[i]],
                                       fontsize=fontsize)

    for n in range(N):
        # avoid overlapping lowest and highest ticks mark
        # print "setting x and y tick freq for {0}".format((n, n))
        ax2 = axarr[n, n]
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))

    # print "setting x and y tick freq for {0}".format((i, j))
    for i in range(N):
        for j in range(N):  # range(i)
            ax2 = axarr[i, j]
            ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
            ax2.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))

    # rotate the xlabels appropriately
    if xlabel_to_rot is not None:
        match_ix = [var_list.index(item) for item in var_list]
        # ok to use for-loops for small number of iterations
        for ix in match_ix:
            labels = axarr[N - 1, ix].get_xticklabels()
            for label in labels:
                label.set_rotation(xlabel_to_rot[var_list[ix]])

    # start plotting the diagonal
    for i in range(N):
        print "N_bins = {0}".format(N_bins[var_list[i]])
        histplot1d_part(axarr[i, i], np.array(data[var_list[i]]),
                        np.array(data['prob']),
                        N_bins=N_bins[var_list[i]],
                        histrange=histran[var_list[i]],
                        x_lim=axlims[var_list[i]])

    # start plotting the lower triangle when row no > col no
    for i in range(N):
        for j in range(i):
            histplot2d_part(axarr[i, j], np.array(data[var_list[j]]),
                            np.array(data[var_list[i]]),
                            prob=np.array(data['prob']),
                            N_bins=Nbins_2D[(var_list[j], var_list[i])],
                            x_lim=axlims[var_list[j]],
                            y_lim=axlims[var_list[i]])

    if save:
        print "saving plot to {0}".format(path + prefix + suffix)
        plt.savefig(path + prefix + suffix, dpi=200, bbox_inches='tight')

    return



