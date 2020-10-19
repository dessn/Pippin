# CosmoMC templates

This directory may be overwhelming if you are new to CosmoMC. Let us break it down:

1. **ini files**: Read in my CosmoMC to configure the runs. You will have a bunch of these for different
cosmological models, combinations of datasets, and different sets of systematics (indicated by the index at the 
end of the ini files).
2. **data.txt**: This file will contain the distances and the independent statistical errors.
3. **sys_\*.txt**: These files will contain flattened covariance matrices that CosmoMC will add on top of the diagonal
error provided in the dataset.
4. **dataset.txt**: Files that combine the data and the covariance matrix, essentially bundling them up together.

