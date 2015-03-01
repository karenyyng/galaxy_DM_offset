# Analysis of galaxy DM offset in the Illustris simulation  
Author: Karen Ng 

To convert the md to pdf:
```sh
$ pandoc -V geometry:margin=1in -o proposed_offset_tests.pdf proposed_offset_tests.md
```

# Proposed tests for characterizing offsets of the mode of the galaxies and the DM  

## Parametric test - the unequal variance test for $H_0: \bar{X} - \bar{Y} = 0$ 
Reference : [p. 294 of Applied Multivariate Statistical Analysis - an
Approximation to the distribution of $T^2$](https://www.dropbox.com/s/sl3sniftx5j10k6/Applied%20Multivariate%20Statistical%20Analysis%20-%20Johnson%20R.%2C%20Wichern%20D_.pdf?dl=0)  
We represent the mode (density peak) of the galaxies as $\bar{X}$ and the mode of the
dark matter (DM) as $\bar{Y}$ both $\bar{X}$ and $\bar{Y}$ are the
modes, the Central Limit Theorem should ensure some degree of
normality.    

To find the confidence regions of the modes, we   

1. bootstrap the identified member subhalos for both the $X$ and the $Y$, 
4. compute the variance-covariance matrices $\Sigma_{\bar{X}}$ and
$\Sigma_{\bar{Y}}$ 
5. test for normality for each bootstrapped distribution of $\bar{X}$ and
$\bar{Y}$ 
6. if there is no strong normality violation^{a},
7. a way to compute suitable the $T^2$ test statistic is: 
$$ t = (\bar{X} - \bar{Y} - (\mu_{\bar{X}} - \mu_{\bar{Y}}))^T S^{-1}
(\bar{X} - \bar{Y} - (\mu_{\bar{X}} - \mu_{\bar{Y}}))$$ , 
with $S = 1 /n_X S_{\bar{X}} +  1 / n_Y S_{\bar{Y}}$ and $S$ being the
sample covariance matrix, and $E(\bar{X} - \bar{Y}) = (\mu_{\bar{X}} - \mu_{\bar{Y}})$

Or can look at:
 
* the [Hotelling package](http://cran.r-project.org/web/packages/Hotelling/Hotelling.pdf)
* the [ICSNP](http://cran.r-project.org/web/packages/ICSNP/index.html)   
It is not obvious to me which of the functions in each package take into account
different sample sizes and different variance assumptions . . . 
The T^2 statistic seem simple enough I may just write my own functions to do this. 

## question 
a. How worried should I be about normality violation? Should I explicitly check for this? 

b. With the different spatial projections (x-z, y-z, and x-y), how do we generalize the t-statistic to
'higher dimension'? (I was not worried about vector algebra at all) To be more
specific, it seems like I will have two estimates for the mode
along the x-spatial dimension from the projections x-z and x-y. What if each pairs of the estimates of the
a mode location don't agree with one
another (due to different binning / smoothing of the data for the
different projections)? Do I first compute the t-stat for x-z, y-z and x-y then combine
them? If so there may be discrepancies between the t-stat. It seems to be
simplier to just infer the mode in 3 spatial dimensions to begin with.  

# Proposed methods for finding the mode (to be finalized) 
We employed a kernel density estimation algorithm to compute the smoothed 2D
density distribution of the galaxies in each cluster in 3 orthogonal
projections (x-y, y-z, and z-x).
We performed a smoothed
cross-validation to obtain the optimal (non-diagonal) smoothing
bandwidth matrices ($H$) of the bivariate Gaussian smoothing kernel. 
Specifically, we made use of the statistical package 
[`ks`](http://www.jstatsoft.org/v21/i07) in `R` .
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
I have not decided completely what to do with the other peaks but it is
interesting to find the correlation of the number of peaks with other physical quantities
indicative of merger activity.

# On how to correctly do pair-bootstrapping in the simulation  
* bootstrap the subhalos regardless of stellar content 
* compute $\bar{Y}_i$ for all the subhalos for the i-th bootstrap
* identify the subhalos with enough stellar content and compute
$\bar{X}_i$ for the i-th bootstrap, 
* compute offsets for each of the i-th bootstrap 

# On how the bootstrapping was done in the paper we discussed  
The [link](http://arxiv.org/pdf/1410.2898v1.pdf) to the paper that we talked about  
the relevant section is "7.3 Mass-Galaxy Offset" on page 13. 

After reading it and thinking more carefully, I think the bootstrapping of
the dark matter done by my collaborator may be different from what I have
described. 
In the paper, my collaborator assume halo (spherically symmetric mass density) models
$\rho(r)$ for fitting the
mass of the dark matter. i.e.

mass enclosed $= 4 \pi \int \rho(r) r^2 dr $  

He uses some weird hybrid of MCMC and grid search
that he never explained clearly for the fitting of this halo model 
In the MCMC, he get a bunch of estimates of the centers of the halo from
the chains. I think the bootstrapped centroids are actually from the chain
estimates ... 

My point is, since the lensing analysis is computationally expensive and
the lensing effects are non-local. It is definitely wrong to just
bootstrap the background lensed galaxies and redo the lensing analysis. 

## Action item for me: 
I need to figure out if there is an alternative way to bootstrap the dark
matter density peak without a halo model $rho(r)$ assumption in the real
data... 

# others ...
[Some thoughts on better modeling](https://github.com/karenyyng/PGM-for-sigma_SIDM/blob/master/README.md) 
