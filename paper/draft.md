# Cluster centroids and peak variability from Illustris simulations

# Intro
Why is it important to study the profile and centroid of galaxies?   
DM maps are expensive. We want to:    

* know which galaxy summary statistics will be allow us to stack the
clusters best
* put a lower limit of the variability of galaxy cluster centroid and DM for
cluster that are not going through major mergers  

# Goals 
* characterize the variability (over time) for clusters which can
only be done from a simulation 
* compare with observations 

# Methods  
## sections that need inputs from CfA people 
### properties of simulated clusters 
* Which sets of simulation do we look at - full physics? 
* What format do the simulation data come in? 
* What are the resolutions? (this will limit how precisely we can find the
centroids - how does this translate to observation resolution?) 
* Cluster finder - has someone used a galaxy-cluster finder for the
simulation already ?  

### Basic checks to examine if the clusters look realistic 
* do the DM halos of clusters look like NFW halos, and describe the
concentration
	* triaxiality of DM halos?
* examine number of galaxies in a cluster (down to 5 magnitudes fainter
than the BCG) 
* examine color properties - do we recover the red sequence? 

## Proposed aspects of the snapshots to look at 
* low redshift - redshift between 0.05 and 1.2 not hard lower / upper limit
* 3D quantities
* projected quantities 

### single cluster 
* track properties of the most massive cluster(s) 

### compute stat of individual clusters then compare 
compare 

* clusters in same snapshot within some mass range
* clusters across snapshots 

## proposed quantities to compute / things to examine within a cluster 
* DM density peak  
* galaxy number density centroid / peaks 
* galaxy luminosity centroid (weighted average) / peak (mode) 
	* weighted by rest frame luminosity in  
		* g-i bands (any other suggestions?) 
	* what color cut (red sequence or not) / redshift cuts gives the tightest separation of galaxy centroid from DM peak  
	
* BCG(s) identification 

Let's just call the distance of different galaxy summary stat and the DM
peak as $\delta s$. 

# Results  

## Figures 
* $\delta s$ as a function of time for the most massive cluster(s) (indicating any major merger) 
* $\delta s$ as a function of mass of cluster  
* number of member galaxies as a function of DM mass of clusters 
	* out to r500c (TBD)
	* within 5 magnitude difference from BCG (TBD)    

## Discussion 
* compare to observations - if applicable 

## Other interesting quantities / aspects to examine  
* 3D / direction of $\delta s$ if there is one big merger this might be
interesting to look at 
* Correlation between colors and distance of galaxies from DM centroid 
* how well velocity dispersion constrains the cluster mass 
* X-ray peak (very optional - depends on how easy or hard this is / if the
gas dynamics are realistic enough)
