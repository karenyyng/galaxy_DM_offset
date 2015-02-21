# Analysis of galaxy DM offset in the Illustris simulation  
Author: Karen Ng 

# Proposed methods for finding the mode (to be)
We employed a kernel density estimation algorithm to compute the smoothed 2D
density distribution of the galaxies in each cluster in 3 orthogonal
projections (x-y, y-z, and z-x).
We performed a smoothed
cross-validation to obtain the optimal (non-diagonal) smoothing
bandwidth matrices ($H$) of the bivariate Gaussian smoothing kernel. 
Specifically, we made use of the statistical package 
[`ks`][1] [Duong](http://www.jstatsoft.org/v21/i07) in the R statistical computing environment (R Core Team 2014).
The employed smooth cross-validation procedure eliminates the free
parameters in the KDE and minimizes the asymptotic mean-integrated squared error (AMISE).
After obtaining the KDE estimate, we employed a finite differencing algorithm
to find the local maxima according to the first and second derivatives. 
We sorted the local maxima according to the corresponding KDE
densities at the maxima and identified the dominant peak. 

## caveat  
The number of peaks found for a given cluster is somewhat indicative of
merger activity. I am aware of the problem that the cluster recently
undergone a merger may better be
modeled by several components. But I do not want to limit myself to
modeling an integer number of components so I chose to do a KDE instead.  

Merger activity is also one of the reasons why I argue that a simple
computation of the mean (or
any variant thereof) is not a good point estimate. 

## question
What if the corresponding peak
positions do not match up in the relevant pairs of projections . . . e.g.
x-positions of the peaks match up in x-y and z-x projections?
If I just do the KDE for the 3 spatial dimensions simultaneously, 
I do not have to worry about this and I am not sure I want to model the
projection uncertainty. Astronomers should just multiply
$\cos(\theta)$ with $\theta$ being the projection
angle to get the probability of seeing the projected version of the
offset by themselves $\pm$ modeling uncertainty.

# Proposed tests for characterizing offsets of the mode of the galaxies and the DM  

## Parametric test - the unequal variance test for $H_0: \bar{X} - \bar{Y} = 0$ 
Reference : [p. 294 of Applied Multivariate Statistical Analysis - an
Approximation to the distribution of $T^2$](https://www.dropbox.com/s/sl3sniftx5j10k6/Applied%20Multivariate%20Statistical%20Analysis%20-%20Johnson%20R.%2C%20Wichern%20D_.pdf?dl=0)  
We represent the mode (density peak) of the galaxies as $\bar{X}$ and the mode of the
dark matter (DM) as $\bar{Y}$ both $\bar{X}$ and $\bar{Y}$ are the
mode, the Central Limit Theorem should ensure some degree of
normality.    

To find the confidence regions of the modes, we   

1. bootstrap the identified member subhalos of a cluster regardless of the
amount of stellar mass for both the $X$ and the $Y$, 
4. compute the variance-covariance matrices $\Sigma_{\bar{X}}$ and
$\Sigma_{\bar{Y}}$ 
5. test for normality for each bootstrapped distribution of $\bar{X}$ and
$\bar{Y}$ 
6. if there is no strong normality violation,
7. a way to compute suitable the $T^2$ test statistic is: 
$$ t = (\bar{X} - \bar{Y} - (\mu_{\bar{X}} - \mu_{\bar{Y}}))^T S^{-1}
(\bar{X} - \bar{Y} - (\mu_{bar_{X}} - \mu_{\bar{Y}}))$$ , 
with $S = 1 /n_X S_{\bar{X}} +  1 / n_Y S_{\bar{Y}}$ and $S$ being the
sample covariance matrix, and $E(\bar{X} - \bar{Y}) = (\mu_{\bar{X}} - \mu_{\bar{Y}})$

Or can look at:
 
* the [Hotelling package](http://cran.r-project.org/web/packages/Hotelling/Hotelling.pdf)
* the [ICSNP](http://cran.r-project.org/web/packages/ICSNP/index.html)   
It is not obvious to me which of the functions in each package take into account
different sample sizes and different variance assumptions.
I may just write my own functions to do this. 

## question 
Again, with the different projections, do I just compute 3 t-statistic for
each cluster?
What if two projections out of three agree with no-offset and one doesn't!? 


