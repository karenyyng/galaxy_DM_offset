#----------------unit tests -------------------------------------
source("../ks_KDE.r")

test_do_KDE = function(){
  x <- gaussian_mixture_data(set_seed=T)
  fhat <- do_KDE_and_get_peaks(x) 

  print ("R KDE function ran without error")
}

test_do_KDE()

## -------following tests were replaced new python functions ------------ 
# test_find_peaks_from_2nd_deriv = function(){
#   # I need to adjust the answer that we test against 
#   # also fix this code
#   dens <- rbind(c(1, 1, 1, 1, 1, 1),
#                 c(1, 8, 3, 2, 1, 1),
#                 c(1, 7, 9, 3, 1, 1),
#                 c(3, 2, 4, 4, 5, 2),
#                 c(1, 1, 3, 2, 1, 1),
#                 c(1, 1, 1, 1, 2, 1),
#                 c(1, 1, 1, 1, 1, 1))
# 
#   # there should be fewer peaks after taking into count corner values 
#   peak_row_ix <- as.integer(c(2, 3, 4, 6)) 
#   peak_col_ix <- as.integer(c(2, 3, 5, 5))
# 
#   coords <- find_peaks_from_2nd_deriv(dens) 
#   row_no <- dim(coords)[[1]]
#   stopifnot(identical(as.vector(coords[1:row_no, 1]), peak_row_ix))
#   stopifnot(identical(as.vector(coords[1:row_no, 2]), peak_col_ix))
# 
#   print("Test 1 passed!")
# 
#   coords
# }
# 
# 
# test_do_KDE_with_weights = function(){
#   # @note I don't think `do_KDE_and_get_peaks` works now
#   # it might have been commneted out 
#   # want to test on same data set
#   x <- gaussian_mixture_data(set_seed=T)
#   orig <- c(-2, 2)
#   dist <- apply(x, 1, function(row){ sqrt(sum((row - orig) * (row - orig))) })
#   mask <- dist < .5
#   wt <- rep(1, nrow(x))
# 
#   # make the weight of the points near the left origin to be ridiculous
#   wt[mask] = 20.
#   fhat <- do_KDE_and_get_peaks(x, w=wt) 
#   print ("Test for do_KDE_with_weights ran without error!")
#   # plot(fhat)
# 
# }

