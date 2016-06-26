import matplotlib.pyplot as plt
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
                labelsize=None, legendloc=None, lvls=[68., 95.]):
    """
    :param x: numpy array of data value
    :param ax: matplotlib ax object
    :param prob: numpy array of weights, data array x is weighted by prob.
    :param w
    for the rest of the parameters, see `kde1d`
    """

    # make font looks like latex font
    rc("font", family="serif")
    support, den = kde1d(x, ax, prob=prob, kernel=kernel, bw=bw,
                         fft=fft, gridsize=gridsize, adjust=adjust, cut=cut,
                         clip=clip, xlabel=xlabel, lim=lim,
                         weights=weights,
                         labelsize=labelsize, legendloc=legendloc)

    # we need the 68 percentile
    if 68. not in lvls:
        lvls.append(68.)
    if 95. not in lvls:
        lvls.append(95.)
    alpha_lvl = {68.: 0.5, 95.: 0.2}

    lowCI_ixes = {}
    upCI_ixes = {}
    sum_stat = {}
    # get confidence interval
    for lvl in lvls:
        lowCI_ixes[lvl], upCI_ixes[lvl] = \
            central_CI(support, den, level=lvl, lim=lim)
        sum_stat['low' + str(int(lvl))] = support[lowCI_ixes[lvl]]
        sum_stat['up' + str(int(lvl))] = support[upCI_ixes[lvl]]
        if lvl == 68. or lvl == 95.:
            ax.fill_between(support[lowCI_ixes[lvl]: upCI_ixes[lvl]],
                            den[lowCI_ixes[lvl]: upCI_ixes[lvl]],
                            alpha=alpha_lvl[lvl],
                            color=c)

    loc_ix = lowCI_ixes[68.]
    # compute the location estimate
    loc = C_BI(x)
    # find the density estimate at the location estimate
    while(support[loc_ix] < loc):
        loc_ix += 1
    ylim = ax.get_ylim()
    # xlim = ax.get_xlim()
    ax.axvline(loc, ymin=0.0,
               ymax=(den[loc_ix] + den[loc_ix + 1]) / 2. / ylim[1],
               ls='--', lw=3, c='k')

    ax.set_ylim(0., ylim[1])

    sum_stat['loc'] = loc

    return sum_stat


def histplot1d_part(ax, x, prob=None, N_bins='knuth', histrange=None,
                    x_lim=None, y_lim=None):
    '''
    This take the additional value of an array axes. for use with subplots
    similar to histplot1d but for subplot purposes I believe
    '''
    # compare bin width to knuth bin width
    # if type(N_bins) is int:
    #    print "specified bin width is {0}, Knuth bin size is {1}".format(
    #        N_bins, knuth_N_bins)
    if N_bins == 'knuth':
        binwidth, bins = de.knuth_bin_width(x, return_bins=True)
        knuth_N_bins = bins.size - 1
        N_bins = knuth_N_bins

    hist, binedges, tmp = ax.hist(
        x, bins=N_bins, histtype='step', weights=prob, range=histrange,
        color='k', linewidth=1)

    # Calculate the location and %confidence intervals
    # Since my location and confidence calculations can't take weighted data I
    # need to use the weighted histogram data in the calculations
    for i in np.arange(N_bins):
        if i == 0:
            x_binned = \
                np.ones(hist[i]) * (binedges[i] + binedges[i + 1]) / 2
        elif np.size(x_binned) == 0:
            x_binned = \
                np.ones(hist[i]) * (binedges[i] + binedges[i + 1]) / 2
        else:
            x_temp = \
                np.ones(hist[i]) * (binedges[i] + binedges[i + 1]) / 2
            x_binned = np.concatenate((x_binned, x_temp))
    loc = biweightLoc(x_binned)
    ll_68, ul_68 = bcpcl(loc, x_binned, 1)
    ll_95, ul_95 = bcpcl(loc, x_binned, 2)

    # Create location and confidence interval line plots
    # find the binedge that the location falls into
    # so that the line indicating the location only extends to top of
    # histogram
    loc_ix = find_bin_ix(binedges, loc)
    ll_68_ix = find_bin_ix(binedges, ll_68)
    ul_68_ix = find_bin_ix(binedges, ul_68)
    ll_95_ix = find_bin_ix(binedges, ll_95)
    ul_95_ix = find_bin_ix(binedges, ul_95)

    ax.plot((loc, loc), (0, hist[loc_ix - 1]), ls='--', lw=1, color="k")

    width = binedges[ll_68_ix + 1] - binedges[ll_68_ix]
    for i in range(ll_68_ix, ul_68_ix):
        ax.bar(binedges[i], hist[i], width, lw=0, color="b", alpha=.6)
    for i in range(ll_95_ix, ul_95_ix):
        ax.bar(binedges[i], hist[i], width, lw=0, color="b", alpha=.3)

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    return loc, ll_68, ul_68, ll_95, ul_95


def histplot2d_part(ax, x, y, prob=None, N_bins=100, histrange=None,
                    x_lim=None, y_lim=None):
    '''
    similar to histplot2d
    This take the additional value of an array axes. for use with subplots
    Input:
    x = [1D array of N floats]
    y = [1D array of N floats]
    prefix = [string] prefix of output file
    prob = [None] or [1D array of N floats] weights to apply to each (x,y) pair
    N_bins = [integer] the number of bins in the x and y directions
    histrange = [None] or [array of floats: (x_min,x_max,y_min,y_max)] the range
        over which to perform the 2D histogram and estimate the confidence
        intervals
    x_lim = [None] or [array of floats: (x_min,x_max)] min and max of the range
        to plot
    y_lim = [None] or [array of floats: (x_min,x_max)] min and max of the range
        to plot
    x_label = [None] or [string] the plot's x-axis label
    y_label = [None] or [string] the plot's y-axis label
    legend = [None] or [True] whether to display a legend or not
    '''
    # prevent masked array from choking up the 2d histogram function
    x = np.array(x)
    y = np.array(y)

    # Create the confidence interval plot
    assert prob is not None, "there is no prob given for weighting"

    if histrange is None:
        if prob is not None:
            H, xedges, yedges = \
                np.histogram2d(x, y, bins=N_bins, weights=prob)
        elif prob is None:
            H, xedges, yedges = np.histogram2d(x, y, bins=N_bins)
    else:
        if prob is not None:
            H, xedges, yedges = \
                np.histogram2d(x, y, bins=N_bins,
                                  range=[[histrange[0], histrange[1]],
                                         [histrange[2], histrange[3]]],
                                  weights=prob)
        elif prob is None:
            H, xedges, yedges = np.histogram2d(
                x, y, bins=N_bins, range=[[histrange[0], histrange[1]],
                                          [histrange[2], histrange[3]]])
    H = np.transpose(H)
    # Flatten H
    h = np.reshape(H, (N_bins ** 2))
    # Sort h from smallest to largest
    index = np.argsort(h)
    h = h[index]
    h_sum = np.sum(h)
    # Find the 2 and 1 sigma levels of the MC hist
    for j in np.arange(np.size(h)):
        if j == 0:
            runsum = h[j]
        else:
            runsum += h[j]
        if runsum / h_sum <= 0.05:
            # then store the value of N at the 2sigma level
            h_2sigma = h[j]
        if runsum / h_sum <= 0.32:
            # then store the value of N at the 1sigma level
            h_1sigma = h[j]

    # Create the contour plot using the 2Dhist info
    # define pixel values to be at the center of the bins
    x = xedges[:-1] + (xedges[1] - xedges[0]) / 2
    y = yedges[:-1] + (yedges[1] - yedges[0]) / 2
    X, Y = np.meshgrid(x, y)

    # can use pcolor or imshow to show the shading instead
    ax.pcolormesh(X, Y, H, cmap=pylab.cm.gray_r, shading='gouraud')
    ax.contour(X, Y, H, (h_2sigma, h_1sigma), linewidths=(2, 2),
               colors=((158 / 255., 202 / 255., 225 / 255.),
                       (49 / 255., 130 / 255., 189 / 255.)))

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)


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


def round_to_n_sig_fig(x, n):
    if x > 1:
        return round(x, -int(np.log10(x)) + (n - 1))
    elif x < 1 and x > 0:
        return round(x, -int(np.log10(x)) + (n))
    elif x < 0 and x > -1:
        return -1. * round(np.abs(x), -int(np.log10(np.abs(x)) + (n)))
    elif x < -1:
        return -1. * round(np.abs(x), -int(np.log10(np.abs(x))) + (n - 1))
