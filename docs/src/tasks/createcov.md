# 7. CREATE_COV

Assuming the biascor task hasn't died, its time to prep for CosmoMC. To do this, we invoke a script from Dan originally (I think) that essentially creates all the input files and structure needed by CosmoMC. It provides a way of scaling systematics, and determining which covariance options to run with.

```yaml
CREATE_COV:
  SOMELABEL:
    MASK: some_biascor_task
    OPTS:
      INI_DIR: /path/to/your/own/dir/of/cosmomc/templates # Defaults to cosmomc_templates, which you can exploit using DATA_DIRS
      SYS_SCALE: surveys/global/lcfit_fitopts/global.yml  # Location of systematic scaling file, same as the FITOPTS file.
      SINGULAR_BLIND: False # Defaults to False, whether different contours will have different shifts applied
      BINNED: True  # Whether to bin the SN or not for the covariance matrx. Defaults to True
      REBINNED_X1: 2 # Rebin x1 into 2 bins
      REBINNED_C: 4 # Rebin c into 4 bins
      SUBTRACT_VPEC: False # Subtract VPEC contribution to MUERR if True. Used when BINNED: False
      FITOPT_SCALES:  # Optional
        FITOPT_LABEL: some_scale  # Note this is a partial match, ie SALT2: 1.0 would apply to all SALT2 cal fitopts
       MUOPT_SCALES:
        MUOPT_LABEL: some_scale  # This is NOT a partial match, must be exact
       COVOPTS:  # Optional, and you'll always get an 'ALL' covopt. List format please
          - "[NOSYS] [=DEFAULT,=DEFAULT]"  # This syntax is explained below
```

If you don't specify `SYS_SCALE`, Pippin will search the LCFIT tasks from the BIASCOR dependency and if all LCFIT tasks have the same fitopt file, it will use that.

The `COVOPTS` section is a bit odd. In the square brackets first, we have the label that will be assigned and used in the plotting output later. The next set of square backets is a two-tuple, and it applies to `[fitopts,muopts]` in that order. For example, to get four contours out of CosmoMC corresponding to all uncertainty, statistics only, statistics + calibration uncertainty, and fitopts + C11 uncertainty, we could set:

```yaml
COVOPTS:
  - "[NOSYS] [=DEFAULT,=DEFAULT]"
  - "[CALIBRATION] [+cal,=DEFAULT]"
  - "[SCATTER] [=DEFAULT,=C11]"
```
