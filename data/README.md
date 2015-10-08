# What's in the files 
* `Illustris-1_00135_APillepich_KarenNG_ParticleData_Group_PartType1.h5` - all
    DM particle mass are the same as `6262734.7210382931`

# what you want to look at:
For every run of the actual code that calculates the projection, 
there are two files outputted. 
* a corresponding file with `peak_df` in the `h5` file name - this should
    contain most of the most interesting info 
* one with the word `fhat` in the h5 file name - you can reanalyze the outputs
    from this to extract the projected offsets or query this file to see what
    settings were used to analyze the outputs in the `peak_df` file 

# how are density estimates (fhat) from KDE organized  
if we are looking at the data files with multiple KDE estimates, 
the paths are organized as:
["clstNo", "cut", "weights", "los_axis", "xi", "phi"]
* cuts - minimal - just cut based stellar particles $> 500$ particles and DM
    particles $> 1000$ for us to call something as a galaxy. - what mass does
    this correspond to? 
* los_axis - 0, 1, 2 is x, y, z respective 
