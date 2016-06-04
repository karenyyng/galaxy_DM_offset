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

    ax.plot(support, density, **kwargs)
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


def CI_loc_plot(x, ax, c='b', prob=None, kernel="gau", bw="scott",
                fft=True, gridsize=None, adjust=1, cut=3,
                clip=(-np.inf, np.inf), xlabel=None, lim=None,
                weights=None,
                labelsize=None, legendloc=None, **kwargs):
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
    ax.axvline(loc, ymin=0,
                ymax=den[loc_ix] / ylim[1], ls='--', lw=3, c='k')

    sum_stat = {"loc": loc,
                "low68": support[low68_ix],
                "up68": support[up68_ix],
                "low95": support[low95_ix],
                "up95": support[up95_ix],
                }

    return sum_stat
