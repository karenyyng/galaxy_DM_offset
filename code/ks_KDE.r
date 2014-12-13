# ---------------------------------------------------------------------------
# Kernel density estimates using R and the ks package
# see > vignette("kde", package="ks") for unmodified version of the example 
# Author: Karen Ng <karenyng@ucdavis.edu>
# ---------------------------------------------------------------------------
library(ks)

set.seed(8192)  # comment out if not testing


# ------------ helper functions  ---------------

do_KDE=
  # group most of the steps for performing KDE to minimize the number of
  # functions that I need to wrap in python 
  # @param
  # data = list of floats denoting the data 
  # bandwidth selector = ks.bandwidth_selector object
  # w = list of floats that is same size as data that denotes the weight
  # @return/
  # fhat_pi1 = R object from the ks package  
function(data, bandwidth_selector, w=rep.int(1, nrow(data))){
  H <- bandwidth_selector(x=data)
  fhat_pi1 <- kde(x=data, H=H, w=w) 
}


find_peaks_from_2nd_deriv= 
  # do numerical differentiation to find peaks 
  # @params
  # fhat = object returned by ks.KDE 
  # return_peak_ix = bool, whether to return R index or actuall coords
  # @return 
  # list with pairs of coordinates indices for the peak values  
  # want to change this to the actual coords
  # @note can consider rewriting this using a faster language than R ...
  # @stability passed test case 
function(fhat, verbose=F, return_peak_ix=T)
{
  dens <- as.matrix(fhat$estimate)
  add_row <- c(rep(0, each=dim(dens)[[2]]))
  add_col <- c(rep(0, each=dim(dens)[[1]]))

  # find peaks along each column
  # take diff, note the sign of change, 
  # bind row, take 2nd diff
  ix_col_peaks <- diff(rbind(add_row, sign(diff(dens))))

  # bind row to preserve dimensionality
  ix_col_peaks <- rbind(ix_col_peaks, add_row)

  # find peaks along each row 
  # tranposed during diff since the diff function only take differences of rows 
  ix_row_peaks <- diff(rbind(add_col, sign(diff(t(dens)))))
  # bind row to preserve dimensionality
  ix_row_peaks <- rbind(ix_row_peaks, add_col) 
  # transpose back to original orientation
  ix_row_peaks <- t(ix_row_peaks)

  if(verbose)
  {
    print(ix_row_peaks)
    print(ix_col_peaks + ix_row_peaks == -4)
    print(which(ix_col_peaks + ix_row_peaks == -4, arr.ind=T))
  }

  if (return_peak_ix)
  {
    return(which(ix_col_peaks + ix_row_peaks == -4, arr.ind=T)) 
  } else
  {
    xloc <- fhat$eval.points[[1]]
    yloc <- fhat$eval.points[[2]]
    return(lapply(1:dim(coord_ix)[[1]], function(i){
           c(xloc[[coord_ix[[i, 1]]]], yloc[[coord_ix[[i, 2]]]])}))
  }
}


find_dominant_peaks=
  # find dominant peaks 
  # @params
  # fhat = R object generated from ks.KDE 
  # coords = list of floats, the floats represent the coordinates  
  # dom_peak_no = integers, number of dominant peaks to find 
  # @stability 
  # it runs without the world crashing and burning but use with caution
function(fhat, coords, dom_peak_no=2L)
{
  # should indicate how many peaks were found 
  print(sprintf("Total num. of peaks found: %d", dim(coords)[[1]]))

  dens <- fhat$estimate[coords] 
  xloc <- fhat$eval.points[[1]]
  yloc <- fhat$eval.points[[2]]

  # sort the array in descending order
  sorted_dens <- sort(dens, decreasing=T)

  # find the peak locs
  peak_locs <- lapply(1:dom_peak_no,
               function(i) c(xloc[coords[which(dens == sorted_dens[[i]],
                                               arr.ind=T), 1]],
                             yloc[coords[which(dens == sorted_dens[[i]],
                                               arr.ind=T), 2]])) 
}


TwoDtestCase1 = 
function(samp_no = 5e2, cwt = 1 / 11)
  # test case with 3 normal mixtures 
  # @param 
  # samp_no = integer, number of data points to draw 
  # cwt = float, 
  #       weight for the central normal mixture that acts as contaminant  
{ # make fake data as 3 normal mixtures  
  mu_s <- rbind(c(-2, 2), c(0, 0), c(2, -2))

  # not sure why the estimated main gaussians are squashed 
  Sigma_s <- rbind(diag(2), 
                   matrix(c(0.8, -0.72, -0.72, 0.8), nrow=2),
                   diag(2))

  # the weights in front of each mixture have to be sum to 1
  n_cwt = (1 - cwt) / 2
  weights <- c(n_cwt, cwt, n_cwt)
  
  # draw data
  x <- rmvnorm.mixt(n=samp_no, mus=mu_s, Sigmas=Sigma_s, props=weights)
  
  # use bandwidth selector either Hpi or Hscv or replace with Hscv 
  do_KDE(x, Hscv)
}


plot_KDE_peaks=
  # get the parameters that we want
  # @param fhat_pi1 = object returned by ks.KDE 
  # @param plot = bool 
  # @param plot_name = str
function(fhat_pi1, plot=T, cf_lvl=c(1:4 * 20.), plot_name="./plots/R_KDE_plot.png",
         save=F)
{ 
  coords_ix <- find_peaks_from_2nd_deriv(fhat_pi1) 
  peaks <- find_dominant_peaks(fhat_pi1, coords_ix)

  # activate the png device 
  if(save) png(plot_name)

  plot(fhat_pi1, cont=cf_lvl, xlab="x", ylab="y")
  for(i in 1:length(peaks)){
    points(peaks[[i]][1], peaks[[i]][2], col="red", pch=20)
  }
  title("R KDE contour plot")

  # output plot from the device
  if(save) dev.off(); system(paste("open", plot_name))
 
  peaks
}


#bootstrap=
#  # DON"T USE
#  # perform bootstrapping for getting confidence regions 
#function(fhat, bootNo=10)
#{ 
#  nrows <- dim(fhat$x)[[1]]
#
#  for(i in 1:bootNo){
#    do_KDE(fhat$x[sapply(runif(nrows, min=1, max=nrows), round), 1:2], Hscv)
#  }
#}
