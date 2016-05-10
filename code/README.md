# Usage info for `get_KDE.py` and `ks_KDE.r`
You will have to make a soft link to `ks_KDE.r` to a particular subdirectory 
if you want to work in a subdirectory of this directory.
```
$ ln -s ../ks_KDE.r .
```

# Folder organization 
* python modules are placed at current directory
* `analyses` contains scripts / notebooks that generate plots / results for paper
* `prototypes` contains code for me to see if my module functions work or not 
* `EDA` contains some exploratory data analyses that investigates aspects of
Illustris data 
* `tests` contain unit tests / integration tests 

## Python modules 
* `compute_distance.py` - helper functions for analyzing the outputs after
    feature engineering 


# Diagnostics of whether your `ks` R package installation works
```
$ Rscript ks_KDE.r  # checks if R and R packages was installed properly
```
