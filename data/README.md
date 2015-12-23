# What's in the files 
* `Illustris-1_00135_APillepich_KarenNG_ParticleData_Group_PartType1.h5` - all
    DM particle mass are the same as `6262734.7210382931`
* `Illustris-1_00135_APillepich_KarenNG_ParticleData_Group_PartType4.h5` - all
    stellar particles 
* `Illustris-1_fof_subhalo_myCompleteHaloCatalog_00135.hdf5` - all the subhalos
    in Illustris, there is an ID to identify to which Cluster / Group each
    subhalo belongs to. 

# Which file to look at for computing offsets      
* `test_peak_df_129.h5`    
    * contains peak info: coordinates, indices in KDE grid, peak density etc. 
    * contains metadata for the run: cuts, weights etc.
* `test_fhat_129.h5`    
    * contains the KDE density maps - this file is big

# How to open the peak_df
More convenient way, just use `pd.read_hdf`:      
```Python
import pandas as pd  
df = pd.read_hdf(PATH_TO_PEAK_DF_FILE, KEY)  # KEY='peak_df' in test runs
```

Not so convient way but you can figure out the hdf5 group key:
```Python
import pandas as pd  
store = pd.HDFStore(PATH_TO_FILE)
store.keys()  # this gives you the key that corresponds to the dataframe (df)
```
