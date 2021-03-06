# ---------------------------------------------------------------------------
# Kernel density estimates using R and the ks package
# see > vignette("kde", package="ks") for unmodified version of the example 
# Author: Karen Ng <karenyng@ucdavis.edu>
# ---------------------------------------------------------------------------
library(ks)
library(parallel)

do_KDE = function(data, bandwidth_selector=Hscv, w=rep.int(1, nrow(data)), 
                  verbose=F, dom_peak_no=1L){
  # group most of the steps for performing KDE to minimize the number of
  # functions that I need to wrap in python 
  # @param
  # data = vector of floats denoting the data 
  # bandwidth selector = ks.bandwidth_selector object
  # w = vector of floats that is same size as data that denotes the weight
  # @return
  # fhat_pi1 = R object from the ks package  
  if (length(dim(data)) == 2){
    H <- bandwidth_selector(x=data)
  } else if (length (dim(data)) == 0){
    # a vector has NULL dimension and gives 0 in the above
    # condition
    H <- hscv(data)
  }
  fhat_pi1 <- kde(x=data, H=H, w=w, binned = FALSE) 

  return(fhat_pi1)
}

gaussian_mixture_data = function(samp_no = 5e2, cwt = 1 / 11, set_seed = FALSE){
  # this prepares samples from 3 gaussian mixtures for test purpose
  # @param sample_no : integer, number of data points to sample in total 
  # @param cwt: fraction < 1 , central contaiminant mixture weight     
  # @param set_seed : boolean, whether to set random seed for reproducibility 
  # @returns fhat R object from ks.KDE method

  if(set_seed){
    seed <- 8192
    set.seed(seed)  
    # needs to `print` if called from python not just sprintf...
    print(sprintf("seed set to %d", seed))
  }

  # make fake data as 3 normal mixtures  
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
}

TwoDtestCase1 = function(samp_no = 5e2, cwt = 1 / 11)
{ 
  # test case with 3 normal mixtures 
  # @param 
  # samp_no = integer, number of data points to draw 
  # cwt = float, 
  #       weight for the central normal mixture that acts as contaminant  
  x <- gaussian_mixture_data(samp_no, cwt) 

  # use bandwidth selector either Hpi or Hscv or replace with Hscv 
  do_KDE(x, Hscv)
}

bootstrap_KDE = function(data_x, bootNo=4L, nrows=nrow(data_x), ncpus=2L, 
                         dom_peak_no=1L,
                         bw_selector=Hscv, w=rep.int(1, nrow(data_x))){ 
  # perform bootstrapping for getting confidence regions 
  # @param data_x:  matrix 
  # @param bootNo: integer 
  # etc.
  # @stability : needs much more debugging to see how the results are stacked
  # and returned
  cl <- makeCluster(ncpus, "FORK")
  ix_list <- lapply(1:bootNo, function(i) 
                    ix <- sample(1:nrows, nrows, replace=T))
  res <- parSapply(cl, ix_list, 
                   function(ix) do_KDE_and_get_peaks(data_x[ix, 1:2], 
                                                     bw_selector,
                                                     w=w,
                                                     dom_peak_no=dom_peak_no))
  stopCluster(cl)
  ix_list <- NULL  # delete the ix list 
  gc()  # tell R to collect memory from the deleted variables

  return(t(res))
}

# ----- I broke some of the functions / not stable for the script below---

# find_peaks_from_2nd_deriv = function(fhat, verbose=F, return_peak_ix=T){
#   # do numerical differentiation to find peaks 
#   # @params
#   # fhat = object returned by ks.KDE 
#   # return_peak_ix = bool, whether to return R index or actual coords
#   #
#   # @return 
#   # list with pairs of coordinates indices for the peak values  
#   # want to change this to the actual coords
#   #
#   # @note can consider rewriting this using a faster language than R ...
#   # @WARNING currently does not compare against peak values 
#   dens <- as.matrix(fhat$estimate)
#   add_row <- c(rep(0, each=dim(dens)[[2]]))
#   add_col <- c(rep(0, each=dim(dens)[[1]]))
# 
#   # find peaks along each column
#   # take diff, note the sign of change, 
#   # bind row, take 2nd diff
#   ix_col_peaks <- diff(rbind(add_row, sign(diff(dens))))
# 
#   # bind row to preserve dimensionality
#   ix_col_peaks <- rbind(ix_col_peaks, add_row)
# 
#   # find peaks along each row 
#   # tranposed during diff since the diff function only take differences of rows 
#   ix_row_peaks <- diff(rbind(add_col, sign(diff(t(dens)))))
#   # bind row to preserve dimensionalityt
#   ix_row_peaks <- rbind(ix_row_peaks, add_col) 
#   # transpose back to original orientation
#   ix_row_peaks <- t(ix_row_peaks)
# 
#   if(verbose)
#   {
#     print(ix_row_peaks)
#     print(ix_col_peaks + ix_row_peaks == -4)
#     print(which(ix_col_peaks + ix_row_peaks == -4, arr.ind=T))
#   }
# 
#   if (return_peak_ix)
#   {
#     return(which(ix_col_peaks + ix_row_peaks == -4, arr.ind=T)) 
#   } else
#   {
#     xloc <- fhat$eval.points[[1]]
#     yloc <- fhat$eval.points[[2]]
#     return(lapply(1:dim(coord_ix)[[1]], function(i){
#            c(xloc[[coord_ix[[i, 1]]]], yloc[[coord_ix[[i, 2]]]])}))
#   }
# }


# find_dominant_peaks = function(fhat, coords_ix, dom_peak_no=1L, verbose=T){
#   # find dominant peaks 
#   #
#   # @params
#   # fhat = R object generated from ks.KDE 
#   # coords_ix = list of integers, the integers represent the coordinates  
#   # dom_peak_no = integers, number of dominant peaks to find 
#   #
#   # @returns 
#   # a matrix of coordinates, coordinate ordering is as follows 
#   # each row correspond to each peak coordinates
#   # first column of the matrix corresponds to the x coordinate, 
#   # second col = y coord
#   # it runs without the world crashing and burning but use with caution
#   # should indicate how many peaks were found 
#   if(verbose) print(sprintf("Total num. of peaks found: %d", dim(coords_ix)[[1]]))
# 
#   dens <- fhat$estimate[coords_ix] 
#   xloc <- fhat$eval.points[[1]]
#   yloc <- fhat$eval.points[[2]]
# 
#   # sort the array in descending order
#   sorted_dens <- sort(dens, decreasing=T)
# 
#   # find the peak locs
#   peak_locs <- sapply(1:dom_peak_no,
#                function(i) c(xloc[coords_ix[which(dens == sorted_dens[[i]],
#                                                   arr.ind=T), 1]],
#                              yloc[coords_ix[which(dens == sorted_dens[[i]],
#                                                   arr.ind=T), 2]])) 
#   return(t(peak_locs))
# }


# make_data = 
# function(mu_s = rbind(c(-2, 2), c(0, 0), c(2, -2)), 
#          Sigma_s = rbind(diag(2), 
#                          matrix(c(0.8, -0.72, -0.72, 0.8), nrow=2),         
#                          diag(2)))
# {
# }
 


# plot_KDE_peaks = function(fhat_pi1, cf_lvl=c(1:4 * 20.), 
#                           plot_name="./plots/R_KDE_plot.png",
#                           save=F, open=F, dom_peak_no=1L){ 
#   # get the parameters that we want
#   # @param fhat_pi1 = object returned by ks.KDE 
#   # @param plot = bool 
#   # @param plot_name = str
#   # @param dom_peak_no = integer 
#   coords_ix <- find_peaks_from_2nd_deriv(fhat_pi1) 
#   peaks <- find_dominant_peaks(fhat_pi1, coords_ix, dom_peak_no=dom_peak_no)
# 
#   # activate the png device 
#   if(save) png(plot_name)
# 
#   plot(fhat_pi1, cont=cf_lvl, xlab="x", ylab="y")
#   for(i in 1:dim(peaks)[[1]]){
#     points(peaks[[i, 1]], peaks[[i, 2]], col="red", pch=20)
#   }
#   title("R KDE contour plot")
# 
#   # output plot from the device
#   if(save){
#     dev.off()
# 
#     # open up the saved plot 
#     if(open) system(paste("open", plot_name))
#   }
#  
#   peaks
# }


#----------------------DANGER UNTESTED UNSTABLE ZONE-----------------------

# do_KDE_and_get_peaks=
#   # perform KDE then get peaks 
#   # @param x: matrix, 2D matrix for holding data values   
#   # @param bw_selector: ks object called bandwidth_selector 
#   # @param w: vector of floats 
#   # 
#   # @return results: list of peaks and fhat
#   # @stability : seems ok - a little ad hoc  
# function(x, bw_selector=Hscv, w=rep.int(1, nrow(x)), 
#          dom_peak_no=1L) 
# {
#   fhat_pi1 <- do_KDE(x, bandwidth_selector=bw_selector, w=w)
#   coords_ix <- find_peaks_from_2nd_deriv(fhat_pi1) 
#   peaks <- find_dominant_peaks(fhat_pi1, coords_ix, dom_peak_no=dom_peak_no)
# 
#   fhat_pi1$coords_ix <- coords_ix
#   fhat_pi1$domPeaks <- peaks
#   return(fhat_pi1)
# }



 
# plot_bootst_KDE_peaks = function(bt_peaks, truth=NULL){
#   # rough draft of how we can plot the peaks
#   # params bt_peaks = vector from bootstrap_KDE function
#   
#   # assuming that we really just select the first 2 peaks
#   plot(rbind(t(bt_peaks[1:2,]), t(bt_peaks[3:4, ])), 
#        xlim=c(-6, 6), ylim=c(-6, 6), xlab='x', ylab='y') 
#   title('bootstrapped peak locations')
# }
# 
# sort_peaks = function(peaks){
#   # sort according to the x coordinate  
#   peaks <- peaks[, sort(peaks[1,], index.return=T)$ix]  
# }


