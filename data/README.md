# What's in the files 
* `Illustris-1_00135_APillepich_KarenNG_ParticleData_Group_PartType1.h5` - all
    DM particle mass are the same as `6262734.7210382931`

# how are density estimates (fhat) from KDE organized  
if we are looking at the data files with multiple KDE estimates, 
the paths are organized as:
["clstNo", "cut", "weights", "los_axis", "xi", "phi"]
* cuts - minimal - just cut based stellar particles $> 500$ particles and DM
    particles $> 1000$ for us to call something as a galaxy. - what mass does
    this correspond to? 
* los_axis - 0, 1, 2 is x, y, z respective 
