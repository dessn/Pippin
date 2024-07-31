# Bias Corrections

With all the probability goodness now in the FITRES files, we can move onto calculating bias corrections. For spec-confirmed surveys, you only need a Ia sample for bias corrections. For surveys with contamination, you will also need a CC only simulation/lcfit result. For each survey being used (as we would often combine lowz and highz surveys), you can specify inputs like below.

```yaml
BIASCOR:
  LABEL:
    # The base input file to utilise
    BASE: surveys/des/bbc/bbc.input
    
    # The names of the lcfits_data/simulations going in. List format please. Note LcfitLabel_SimLabel format
    DATA: [DESFIT_DESSIM, LOWZFIT_LOWZSIM]
    
    # Input Ia bias correction simulations to be concatenated
    SIMFILE_BIASCOR: [DESFIT_DESBIASCOR, LOWZFIT_LOWZBIASCOR]

    # Optional, specify FITOPT to use. Defaults to 0 for each SIMFILE_BIASCOR. If using this option, you must specify a FITOPT for each SIMFILE_BIASCOR
    SIMFILE_BIASCOR_FITOPTS: [0, 1] # FITOPT000 and FITOPT001

    # For surveys that have contamination, add in the cc only simulation under CCPRIOR    
    SIMFILE_CCPRIOR: DESFIT_DESSIMBIAS5YRCC

    # Optional, specify FITOPT to use. Defaults to 0 for each SIMFILE_CCPRIOR. If using this option, you must specify a FITOPT for each SIMFILE_CCPRIOR
    SIMFILE_CCPRIOR_FITOPTS: [0, 1] # FITOPT000 and FITOPT001


    # Which classifier to use. Column name in FITRES will be determined from this property.
    # In the case of multiple classifiers this can either be
    #    1. A list of classifiers which map to the same probability column name (as defined by MERGE_CLASSIFIERS in the AGGREGATION stage)
    #    2. A probability column name (as defined by MERGE_CLASSIFIERS in the AGGREGATION stage)
    # Note that this will crash if the specified classifiers do not map to the same probability column.
    CLASSIFIER: UNITY
    
    # Default False. If multiple sims (RANSEED_CHANGE), make one or all Hubble plots.
    MAKE_ALL_HUBBLE: False
    
    # Defaults to False. Will load in the recalibrated probabilities, and crash and burn if they dont exist.
    USE_RECALIBRATED: True
    
    # Defaults to True. If set to True, will rerun biascor twice, removing any SNID that got dropped in any FITOPT/MUOPT
    CONSISTENT_SAMPLE: False

  
  # We can also specify muopts to add in systematics. They share the structure of the main biascor definition
  # You can have multiple, use a dict structure, with the muopt name being the key
  MUOPTS:
      C11:
        SIMFILE_BIASCOR: [D_DESBIASSYS_C11, L_LOWZBIASSYS_C11]
        SCALE: 0.5 # Defaults to 1.0 scale, used by CREATE_COV to determine covariance matrix contribution
        
  # Generic OPTS that can modify the base file and overwrite properties
  OTPS:
    BATCH_INFO: sbatch $SBATCH_TEMPLATES/SBATCH_Midway2_1hr.TEMPLATE 10
```

For those that generate large simulations and want to cut them up into little pieces, you want the `NSPLITRAN` syntax. The configuration below will take the inputs and divide them into 10 samples, which will then propagate to 10 CosmoMC runs if you have a CosmoMC task defined.

```yaml
BIASCOR:
  LABEL:
    BASE: surveys/des/bbc/bbc_3yr.input
    DATA: [D_DES_G10]
    SIMFILE_BIASCOR: [D_DESSIMBIAS3YRIA_G10]
    PROB_COLUMN_NAME: some_column_name  # optional instead of CLASSIFIER
    OPTS:
      NSPLITRAN: 10
```
