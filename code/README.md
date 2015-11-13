# Usage info for `get_KDE.py` and `ks_KDE.r`
You will have to make a soft link to `ks_KDE.r` to a particular subdirectory 
if you want to work in a subdirectory of this directory.
```
$ ln -s ../ks_KDE.r .
```

# File organization 
* python modules are placed at this directory
* `analyses` contains scripts / notebooks that generate plots for paper
* `EDA` are some exploratory data analyses that investigates aspects of
Illustris data 
* `prototypes` are code for me to see if my module functions work or not 
* `tests` contain mainly working unit tests / tests that need to be implemented properly

# diagnostics of whether your installation works
```
$ Rscript ks_KDE.r  # checks if R and R packages was installed properly
```
