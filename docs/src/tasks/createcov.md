# 7. CREATE_COV
 
After BBC/BIASCOR stage, its time to use "create_covariance.py" to prep sys-covariance and covtot^-1 matrices for COSMOFIT stage

```yaml
CREATE_COV:
  SOMELABEL:
    MASK: some_biascor_task
    OPTS:
      SYS_SCALE: surveys/global/lcfit_fitopts/global.yml  # Location of sys scaling file, same as the FITOPTS file.
      SINGULAR_BLIND: False # Defaults to False, whether different contours will have different shifts applied
      BINNED: True  # Whether to bin the SN or not for the covariance matrx. Defaults to True
      REBINNED_X1: 2 # Rebin x1 into 2 bins
      REBINNED_C:  4 # Rebin c into 4 bins
      SUBTRACT_VPEC: False # Subtract VPEC contribution to MUERR if True. Used when BINNED: False
      FITOPT_SCALES:  # Optional
        FITOPT_LABEL: some_scale  # Note this is a partial match, ie SALT2: 1.0 would apply to all SALT2 cal fitopts
      MUOPT_SCALES:
        MUOPT_LABEL: some_scale  # This is NOT a partial match, must be exact
      COVOPTS:  # Optional, and you always get an 'ALL' covopt. List format please
          - "[NOSYS] [=DEFAULT,=DEFAULT]"  
          - etc ...                         get more help with  "create_covariance.py -H"

      EXTRA_OPTS:  [optional: any other command line options from create_covariance.py -h]
```

If you don't specify `SYS_SCALE`, Pippin will search the LCFIT tasks from the BIASCOR dependency and if all LCFIT tasks have the same fitopt file, it will use that.

