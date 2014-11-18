# ----------------------------------------------------------------------
# trying out functions in ks 
# see vignette("kde", package="ks") for unmodified version of the example 
# ----------------------------------------------------------------------
library(ks)

set.seed(8192)  # comment out if not testing
samp_no <- 1e3

# ------------ helper functions  ---------------
find_ix = 
function(dens, id_ix)
{
  shape <- sqrt(length(dens))
  row_no <- round(id_ix / shape)
  col_no <- id_ix %% shape

  return (c(row_no, col_no))
}


find_peaks = 
function(dens)
{

}


TwoDtestCase1 = 
# test case with 3 normal mixtures 
function(){
  # make fake data as 3 normal mixtures  
  mu_s <- rbind(c(-2, 2), c(0, 0), c(2, -2))
  Sigma_s <- rbind(diag(2), 
                   matrix(c(0.8, -0.72, -0.72, 0.8), nrow=2),
                   diag(2))

  # the weights in front of each mixture have to be sum to 1
  cwt <- 1. / 11  # the central mixture weight 
  weights <- c((1 - cwt) / 2, cwt, (1 - cwt) / 2)
  
  # draw data
  x <- rmvnorm.mixt(n=samp_no, mus=mu_s, Sigmas=Sigma_s, props=weights)
  plot(x)
  
  # use bandwidth selector or replace with Hscv 
  Hpi1 <- Hpi(x=x)  

  # KDE estimate has an option called weight 
  fhat_pi1 <- kde(x=x, H=Hpi1)  
  # plot(fhat_pi1)
  
  ## row major language ... find y coordinate? 
  dens <- unlist(fhat_pi1["estimate"][[1]])
  max_ix <- which.max(dens)
  shape <- sqrt(length(dens))
  find_ix(dens, max_ix)

  # too annoying to deal with the fhat object
  xlocs <- fhat_pi1["eval.points"][[1]][[1]]
  ylocs <- fhat_pi1["eval.points"][[1]][[2]]
  
  # plot to visualize 

}




