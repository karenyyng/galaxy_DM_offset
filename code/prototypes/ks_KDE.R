# ---------------------------------------------------------------------------
# trying out functions in ks 
# see > vignette("kde", package="ks") for unmodified version of the example 
# ---------------------------------------------------------------------------
library(ks)

set.seed(8192)  # comment out if not testing


# ------------ helper functions  ---------------

find_peaks_from_2nd_deriv= 
function(dens, verbose=F)
  # do numerical differentiation to find peaks 
  # @params
  # dens = vector of floats 
  # @return 
  # pairs of coordinates for the peak values  
  # @note can consider rewriting this using a faster language than R ...
{
  dens <- as.matrix(dens)
  add_row <- c(rep(0, each=dim(dens)[[2]]))
  add_col <- c(rep(0, each=dim(dens)[[1]]))

  # find peaks along each column
  # take diff, note the sign of change, 
  # bind rows, take 2nd diff
  ix_col_peaks <- diff(rbind(add_row, sign(diff(dens))))
  ix_col_peaks <- rbind(ix_col_peaks, add_row)

  # find peaks along each row 
  # tranposed during diff since the diff function only take differences of rows 
  ix_row_peaks <- diff(rbind(add_col, sign(diff(t(dens)))))
  ix_row_peaks <- rbind(ix_row_peaks, add_col) 
  ix_row_peaks <- t(ix_row_peaks)

  if(verbose)
  {
    print(ix_row_peaks)
    print(ix_col_peaks + ix_row_peaks == -4)
    print(which(ix_col_peaks + ix_row_peaks == -4, arr.ind=T))
  }

  which(ix_col_peaks + ix_row_peaks == -4, arr.ind=T) 
}


find_dominant_peaks=
function(fhat, coords, dom_peak_no=2L)
  # find dominate peaks 
{
  # should indicate how many peaks were found 
  # not sure why nothing is printing
  sprintf("Total no of peaks found %d", dim(coords)[[1]])

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
 
  #diff <- lapply(coords[c(2: length(coords))],
  #               function(x, mcoord) sqrt(sum((x - mcoord) ** 2)),
  #               coords[[1]])  
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
  
  # use bandwidth selector or replace with Hscv 
  #Hpi1 <- Hpi(x=x) 
  Hscv1 <- Hscv(x=x)

  # KDE estimate has an option called weight 
  fhat_pi1 <- kde(x=x, H=Hscv1) 

  fhat_pi1
}
  

do_analysis=
function(fhat_pi1){
  
  coords <- find_peaks_from_2nd_deriv(fhat_pi1$estimate) 
  peaks <- find_dominant_peaks(fhat_pi1, coords)

  # plot to visualize 
  plot(fhat_pi1, cont=c(1, 5, 50, 70))
  points(peaks[[1]][1], peaks[[1]][2], col="red", pch=20)
  points(peaks[[2]][1], peaks[[2]][2], col="red", pch=20)
}




