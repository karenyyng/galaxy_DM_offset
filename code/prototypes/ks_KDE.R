# ----------------------------------------------------------------------
# trying out functions in ks 
# see > vignette("kde", package="ks") for unmodified version of the example 
# ----------------------------------------------------------------------
library(ks)

set.seed(8192)  # comment out if not testing

# ------------ helper functions  ---------------
find_xy_coord_ix = 
function(id_ix, dens)
  # converts id_ix from vector indices into row and col number
{
  shape <- sqrt(length(dens))
  row_no <- round(id_ix / shape)
  col_no <- id_ix %% shape

  return (c(row_no, col_no))
}


find_peaks = 
function(dens, peak_search_no=length(dens))
  # calls a bunch of other functions to find the peaks 
  # @params
  # dens = vector of floats 
  # peak_no = integer, describes how many to look for 
  # @return 
  # pairs of coordinates for the peak values  
  # @note 
  # need a smarter approach to use 2nd derivatives to find local maximas
{
  # sort the array in descending order  
  sorted_dens <- sort(dens, decreasing = T)

  # find the peaks  
  ix <- lapply(1:peak_search_no, 
               function(i) which(dens == sorted_dens[[i]])) 

  coords <- lapply(ix, find_xy_coord_ix, dens)
  diff <- lapply(coords[c(2: length(coords))], 
                 function(x, mcoord) sqrt(sum((x - mcoord) ** 2)),
                 coords[[1]])
}


TwoDtestCase1 = 
function(samp_no = 5e2, cwt = 1 / 11)
  # test case with 3 normal mixtures 
  # @param 
  # samp_no = integer, number of data points to draw 
  # cwt = float, 
  #       weight for the central normal mixture that acts as contaminant  
{
  # make fake data as 3 normal mixtures  
  mu_s <- rbind(c(-2, 2), c(0, 0), c(2, -2))
  Sigma_s <- rbind(diag(2), 
                   matrix(c(0.8, -0.72, -0.72, 0.8), nrow=2),
                   diag(2))

  # the weights in front of each mixture have to be sum to 1
  weights <- c((1 - cwt) / 2, cwt, (1 - cwt) / 2)
  
  # draw data
  x <- rmvnorm.mixt(n=samp_no, mus=mu_s, Sigmas=Sigma_s, props=weights)
  
  # use bandwidth selector or replace with Hscv 
  #Hpi1 <- Hpi(x=x)  
  Hscv1 <- Hscv(x=x)

  # KDE estimate has an option called weight 
  fhat_pi1 <- kde(x=x, H=Hscv1) # H=Hpi1 
  # plot(fhat_pi1)
  fhat_pi1
}
  

do_analysis=
function(fhat_pi1){
  dens <- fhat_pi1$estimate
  max_ix <- which.max(dens)
  shape <- sqrt(length(dens))

  # too annoying to deal with the fhat object with the indices
  xlocs <- fhat_pi1$eval.points[[1]]
  ylocs <- fhat_pi1$eval.points[[2]]
  
  # plot to visualize 
  find_peaks(dens)
  #plot(fhat_pi1, cont=c(1, 5, 50, 70))
  #plot(x)

  list("dens"=dens, "xlocs"=xlocs, "ylocs"=ylocs)
}




