[![Documentation](https://readthedocs.org/projects/pippin/badge/?version=latest)](https://pippin.readthedocs.io/en/latest/?badge=latest)
[![JOSS](https://joss.theoj.org/papers/10.21105/joss.02122/status.svg)](https://doi.org/10.21105/joss.02122)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.366608-blue)](https://zenodo.org/badge/latestdoi/162215291)
[![GitHub license](https://img.shields.io/badge/License-MIT-green)](https://github.com/dessn/Pippin/blob/master/LICENSE)
[![Github Issues](https://img.shields.io/github/issues/dessn/Pippin)](https://github.com/dessn/Pippin/issues)
![Python Version](https://img.shields.io/badge/Python-3.7%2B-red)
![Pippin Test](https://github.com/dessn/Pippin/actions/workflows/test-pippin.yml/badge.svg)

## Tasks

Pippin is essentially a wrapper around many different tasks. In this section, 
I'll try and explain how tasks are related to each other, and what each task is.

As a general note, most tasks have an `OPTS` where most details go. This is partially historical, but essentially properties
that Pippin uses to determine how to construct tasks (like `MASK`, classification mode, etc) are top level, and the Task itself gets passed everything
inside `OPTS` to use however it wants. 

[//]: # (Start of Task specification)

### Data Preparation

The DataPrep task is simple - it is mostly a pointer for Pippin towards an external directory that contains
some photometry, to say we're going to make use of it. Normally this means data files,
though you can also use it to point to simulations that have already been run to save yourself
the hassle of rerunning them.  The other thing the DataPrep task will do is run the new 
method of determining a viable initial guess for the peak time, which will be used by the light curve fitting task down the road. 
The full options available for the DataPrep task are:

```yaml
DATAPREP:
  SOMENAME:
    OPTS:
    
      # Location of the photometry files
      RAW_DIR: $DES_ROOT/lcmerge/DESALL_forcePhoto_real_snana_fits
      
      # Specify which types are confirmed Ia's, confirmed CC or unconfirmed. Used by ML down the line
      TYPES:
        IA: [101, 1]
        NONIA: [20, 30, 120, 130]

      # Blind the data. Defaults to True if SIM:True not set
      BLIND: False
      
      # Defaults to False. Important to set this flag if analysing a sim in the same way as data, as there
      # are some subtle differences
      SIM: False

      # The method of estimating peak mjd values. Don't ask me what numbers mean what, ask Rick.
      OPT_SETPKMJD: 16

```

### Simulation

The simulation task does exactly what you'd think it does. It invokes [SNANA](https://github.com/RickKessler/SNANA) to run some similation as per your configuration. 
If something goes wrong, Pippin tries to dig through the log files to give you a useful error message, but sometimes this
is difficult (i.e. the logs have been zipped up). With the current version of SNANA, each simulation can have at most one Ia component, 
and an arbitrary number of CC components. The specification for the simulation task config is as follows:

```yaml
SIM:
  SOMENAMEHERE:
  
    # We specify the Ia component, so it must have IA in its name
    IA_G10: 
      BASE: surveys/des/sims_ia/sn_ia_salt2_g10_des5yr.input  # And then we specify the base input file which generates it.
      
    # Now we can specify as many CC sims to mix in as we want
    II_JONES:
      BASE: surveys/des/sims_cc/sn_collection_jones.input
    
    IAX:
      BASE: surveys/des/sims_cc/sn_iax.input
      DNDZ_ALLSCALE: 3.0  # Note you can add/overwrite keys like so for specific files

    # This section will apply to all components of the sim
    GLOBAL:
      NGEN_UNIT: 1
      RANSEED_REPEAT: 10 12345
```

### Light Curve Fit

This task runs the SALT2 light curve fitting process on light curves from the simulation or DataPrep task. As above,
if something goes wrong I try and give a good reason why, if you don't get a good reason, let me know. The task is 
specified like so:

```yaml
LCFIT:
  SOMENAMEHERE:
    # MASK means only apply this light curve fitting on sims/Dataprep which have DES in the name
    # You can also specify a list for this, and they will be applied as a logical or
    MASK: DES
      
    # The base nml file used 
    BASE: surveys/des/lcfit_nml/des.nml
      
    # FITOPTS can be left out for nothing, pointed to a file, specified manually or a combination of the two
    # Normally this would be a single entry like global.yml shown below, but you can also pass a list
    # If you specify a FITOPT manually, make sure it has the / around the label
    # And finally, if you specify a file, make sure its a yml dictionary that links a survey name to the correct
    # fitopts. See the file below for an example
    FITOPTS:
      - surveys/global/lcfit_fitopts/global.yml
      - "/custom_extra_fitopt/ REDSHIFT_FINAL_SHIFT 0.0001"

    # We can optionally customise keys in the FITINP section
    FITINP:
      FILTLIST_FIT: 'gri'
      
    # And do the same for the optional SNLCINP section
    SNLCINP:
      CUTWIN_SNRMAX:  3.0, 1.0E8
      CUTWIN_NFILT_SNRMAX:  3.0, 99.

    # Finally, options that go outside either of these sections just go in the generic OPTS
    OPTS:
      BATCH_INFO: sbatch $SBATCH_TEMPLATES/SBATCH_Midway2_1hr.TEMPLATE 10
```

### Classification

Within Pippin, there are many different classifiers implemented. Most classifiers need to be trained, and 
can then run in predict mode. All classifiers that require training can either be trained in the same yml 
file, or you can point to an external serialised instance of the trained class and use that. The general syntax
for a classifier is:

```yaml
CLASSIFICATION:
  SOMELABEL:
    CLASSIFIER: NameOfTheClass
    MODE: train  # or predict
    MASK: mask  # Masks both sim and lcfit together, logical and, optional
    MASK_SIM: sim_only_mask
    MASK_FIT: lcfit_only_mask
    COMBINE_MASK: [SIM_IA, SIM_CC] # optional mask to combine multiple sim runs into one classification job (e.g. separate CC and Ia sims). NOTE: currently not compatible with SuperNNova/SNIRF
    OPTS:
      MODEL: file_or_label  # only needed in predict mode, how to find the trained classifier
      OPTIONAL_MASK: opt_mask # mask for optional dependencies. Not all classifiers make use of this
      OPTIONAL_MASK_SIM: opt_sim_only_mask # mask for optional sim dependencies. Not all classifiers make use of this
      OPTIONAL_MASK_FIT: opt_lcfit_only_mask # mask for optional lcfit dependencies. Not all classifiers make use of this
      WHATREVER_THE: CLASSIFIER_NEEDS  
```

#### SCONE Classifier

The [SCONE classifier](https://github.com/helenqu/scone) is a convolutional neural network-based classifier for supernova photometry. The model first creates "heatmaps" of flux values in wavelength-time space, then runs the neural network model on GPU (if available) to train or predict on these heatmaps. A successful run will produce `predictions.csv`, which shows the Ia probability of each SN. For debugging purposes, the model config (`model_config.yml`), Slurm job (`job.slurm`), log (`output.log`), and all the heatmaps (`heatmaps/`) can be found in the output directory. An example of how to define a SCONE classifier:

```yaml
CLASSIFICATION:
  SCONE_TRAIN: # Helen's CNN classifier
    CLASSIFIER: SconeClassifier
    MODE: train
    OPTS:
      GPU: True # OPTIONAL, default: False
      # HEATMAP CREATION OPTS
      CATEGORICAL: True # OPTIONAL, binary or categorical classification, default: False
      NUM_WAVELENGTH_BINS: 32 # OPTIONAL, heatmap height, default: 32
      NUM_MJD_BINS: 180 # OPTIONAL, heatmap width, default: 180
      REMAKE_HEATMAPS: False # OPTIONAL, SCONE does not remake heatmaps unless the 3_CLAS/heatmaps subdir doesn't exist or if this param is true, default: False
      # MODEL OPTS
      NUM_EPOCHS: 400 # REQUIRED, number of training epochs
      IA_FRACTION: 0.5 # OPTIONAL, desired Ia fraction in train/validation/test sets for binary classification, default: 0.5
  
  SCONE_PREDICT: # Helen's CNN classifier
    CLASSIFIER: SconeClassifier
    MODE: predict
    OPTS:
      GPU: True # OPTIONAL, default: False
      # HEATMAP CREATION OPTS
      CATEGORICAL: True # OPTIONAL, binary or categorical classification, default: False
      NUM_WAVELENGTH_BINS: 32 # OPTIONAL, heatmap height, default: 32
      NUM_MJD_BINS: 180 # OPTIONAL, heatmap width, default: 180
      REMAKE_HEATMAPS: False # OPTIONAL, SCONE does not remake heatmaps unless the 3_CLAS/heatmaps subdir doesn't exist or if this param is true, default: False
      # MODEL OPTS
      MODEL: "/path/to/trained/model" # REQUIRED, path to trained model that should be used for prediction
      IA_FRACTION: 0.5 # OPTIONAL, desired Ia fraction in train/validation/test sets for binary classification, default: 0.5
```

#### SuperNNova Classifier

The [SuperNNova classifier](https://github.com/supernnova/SuperNNova) is a recurrent neural network that
operates on simulation photometry. It has three in vuilt variants - its normal (vanilla) mode, a Bayesian mode
and a Variational mode. After training, a `model.pt` can be found in the output directory,
which you can point to from a different yaml file. You can define a classifier like so:

```yaml
CLASSIFICATION:
  SNN_TEST:
    CLASSIFIER: SuperNNovaClassifier
    MODE: predict
    GPU: True # Or False - determines which queue it gets sent into
    CLEAN: True # Or false - determine if Pippin removes the processed folder to sae space
    OPTS:
      MODEL: SNN_TRAIN  # Havent shown this defined. Or /somepath/to/model.pt
      VARIANT: vanilla # or "variational" or "bayesian". Defaults to "vanilla"
      REDSHIFT: True  # What redshift info to use when classifying. Defaults to 'zspe'. Options are [True, False, 'zpho', 'zspe', or 'none']. True and False are legacy options which map to 'zspe', and 'none' respectively.
      NORM: cosmo_quantile  # How to normalise LCs. Other options are "perfilter", "cosmo", "global" or "cosmo_quantile".  
      CYCLIC: True  # Defaults to True for vanilla and variational model
      SEED: 0  # Sets random seed. Defaults to 0.
      LIST_FILTERS: ['G', 'R', 'I', 'Z'] # What filters are present in the data, defaults to ['g', 'r', 'i', 'z']
      SNTYPES: "/path/to/sntypes.txt" # Path to a file which lists the sn type mapping to be used. Example syntax for this can be found at https://github.com/LSSTDESC/plasticc_alerts/blob/main/Examples/plasticc_schema/elasticc_origmap.txt. Alternatively, yaml dictionaries can be used to specify each sn type individually.
```

Pippin also allows for supernnova input yaml files to be passed, instead of having to define all of the options in the Pippin input yaml. This is done via:

```yaml
OPTS:
    DATA_YML: path/to/data_input.yml
    CLASSIFICATION_YML: path/to/classification_input.yml
```

Example input yaml files can be found [here](https://github.com/supernnova/SuperNNova/tree/master/configs_yml), with the important variation that you must have:

```yaml
raw_dir: RAW_DIR
dump_dir: DUMP_DIR
done_file: DONE_FILE
```

So that Pippin can automatically replace these with the appropriate directories. 

#### SNIRF Classifier

The [SNIRF classifier](https://github.com/evevkovacs/ML-SN-Classifier) is a random forest running off SALT2 summary
statistics. You can specify which features it gets to train on, which has a large impact on performance. After training,
there should be a `model.pkl` in the output directory. You can specify one like so:

```yaml
CLASSIFICATION:
  SNIRF_TEST:
    CLASSIFIER: SnirfClassifier
    MODE: predict
    OPTS:
      MODEL: SNIRF_TRAIN
      FITOPT: some_label  # Optional FITOPT to use. Match the label. Defaults to no FITOPT
      FEATURES: x1 c zHD x1ERR cERR PKMJDERR  # Columns to use. Defaults are shown. Check FITRES for options.
      N_ESTIMATORS: 100  # Number of trees in forest
      MIN_SAMPLES_SPLIT: 5  # Min number of samples to split a node on
      MIN_SAMPLES_LEAF: 1  # Minimum number samples in leaf node
      MAX_DEPTH: 0  # Max depth of tree. 0 means auto, which means as deep as it wants.
```

#### Nearest Neighbour Classifier

Similar to SNIRF, NN trains on SALT2 summary statistics using a basic Nearest Neighbour algorithm from sklearn. 
It will produce a `model.pkl` file in its output directory when trained. You can configure it as per SNIRF:


```yaml
CLASSIFICATION:
  NN_TEST:
    CLASSIFIER: NearestNeighborPyClassifier
    MODE: predict
    OPTS:
      MODEL: NN_TRAIN
      FITOPT: some_label  # Optional FITOPT to use. Match the label. Defaults to no FITOPT
      FEATURES: zHD x1 c cERR x1ERR COV_x1_c COV_x1_x0 COV_c_x0 PKMJDERR  # Columns to use. Defaults are shown.
```

#### Perfect Classifier

Sometimes you want to cheat, and if you have simulations, this is easy. The perfect classifier looks into the sims to 
get the actual type, and will then assign probabilities as per your configuration. This classifier has no training mode,
only predict.

```yaml
CLASSIFICATION:
  PERFECT:
    CLASSIFIER: PerfectClassifier
    MODE: predict
    OPTS:
      PROB_IA: 1.0  # Probs to use for Ia events, default 1.0
      PROB_CC: 0.0  # Probs to use for CC events, default 0.0
```

#### Unity Classifier

To emulate a spectroscopically confirmed sample, or just to save time, we can assign every event a probability of 1.0
that it is a type Ia. As it just returns 1.0 for everything, it only has a predict mode

```yaml
CLASSIFICATION:
  UNITY:
    CLASSIFIER: UnityClassifier
    MODE: predict
```

#### FitProb Classifier

Another useful debug test is to just take the SALT2 fit probability calculated from the chi2 fitting and use that
as our probability. You'd hope that classifiers all improve on this. Again, this classifier only has a predict mode.

```yaml
CLASSIFICATION:
  FITPROBTEST:
    CLASSIFIER: FitProbClassifier
    MODE: predict
```

### Aggregation

The aggregation task takes results from one or more classification tasks (that have been run in predict mode
on the same dataset) and generates comparisons between the classifiers (their correlations, PR curves, ROC curves
and their calibration plots). Additionally, it merges the results of the classifiers into a single
csv file, mapping SNID to one column per classifier.

```yaml
AGGREGATION:
  SOMELABEL:
    MASK: mask  # Match sim AND classifier
    MASK_SIM: mask # Match only sim
    MASK_CLAS: mask # Match only classifier
    RECALIBRATION: SIMNAME # Optional, use this simulation to recalibrate probabilities. Default no recal.
    # Optional, changes the probability column name of each classification task listed into the given probability column name.
    # Note that this will crash if the same classification task is given multiple probability column names.
    # Mostly used when you have multiple photometrically classified samples
    MERGE_CLASSIFIERS:
      PROB_COLUMN_NAME: [CLASS_TASK_1, CLASS_TASK_2, ...]
    OPTS:
      PLOT: True # Default True, make plots
      PLOT_ALL: False # Default False. Ie if RANSEED_CHANGE gives you 100 sims, make 100 set of plots.
```

### Merging

The merging task will take the outputs of the aggregation task, and put the probabilities from each classifier
into the light curve fit results (FITRES files) using SNID.

```yaml
MERGE:
  label:
    MASK: mask  # partial match on all sim, fit and agg
    MASK_SIM: mask  # partial match on sim
    MASK_FIT: mask  # partial match on lcfit
    MASK_AGG: mask  # partial match on aggregation task
```

### Bias Corrections

With all the probability goodness now in the FITRES files, we can move onto calculating bias corrections. 
For spec-confirmed surveys, you only need a Ia sample for bias corrections. For surveys with contamination, 
you will also need a CC only simulation/lcfit result. For each survey being used (as we would often combine lowz and highz
surveys), you can specify inputs like below.

Note that I expect this task to have the most teething issues, especially when we jump into the MUOPTS.

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

For those that generate large simulations and want to cut them up into little pieces, you want the `NSPLITRAN` syntax. 
The configuration below will take the inputs and divide them into 10 samples, which will then propagate to 10 CosmoMC runs
if you have a CosmoMC task defined.

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

### Create Covariance

Assuming the biascor task hasn't died, its time to prep for CosmoMC. To do this, we invoke a script from Dan originally
(I think) that essentially creates all the input files and structure needed by CosmoMC. It provides a way of scaling
systematics, and determining which covariance options to run with.

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

If you don't specify `SYS_SCALE`, Pippin will search the LCFIT tasks from the BIASCOR dependency and if all LCFIT tasks
have the same fitopt file, it will use that.

The `COVOPTS` section is a bit odd. In the square brackets first, we have the label that will be assigned and used
in the plotting output later. The next set of square backets is a two-tuple, and it applies to `[fitopts,muopts]` in 
that order. For example, to get four contours out of CosmoMC corresponding to all uncertainty, statistics only,
statistics + calibration uncertainty, and fitopts + C11 uncertainty, we could set:

```yaml
COVOPTS:
  - "[NOSYS] [=DEFAULT,=DEFAULT]"
  - "[CALIBRATION] [+cal,=DEFAULT]"
  - "[SCATTER] [=DEFAULT,=C11]"
```

### CosmoFit 

CosmoFit is a generic cosmological fitting task, which allows you to choose between different fitters.
The syntax is very simple:
```yaml
COSMOFIT:
    COSMOMC:
        SOMELABEL:
            # CosmoMC options
    WFIT:
        SOMEOTHERLABEL:
            # WFit options
```

#### CosmoMC

Launching CosmoMC is hopefully fairly simple. There are a list of provided configurations under the `cosmomc_templates`
directory (inside `data_files`), and the main job of the user is to pick which one they want. 

```yaml
COSMOFIT:
    COSMOMC:
      SOMELABEL:
        MASK_CREATE_COV: mask  # partial match
        OPTS:
          INI: sn_cmb_omw  # should match the filename of an ini file
          NUM_WALKERS: 8  # Optional, defaults to eight.
          
          # Optional, covopts from CREATE_COV step to run against. If blank, you get them all. Exact matching.
          COVOPTS: [ALL, NOSYS]
```

#### WFit

Launching WFit simply requires providing the command line options you want to use for each fit.
```yaml
COSMOFIT:
    WFIT:
        SOMELABEL:
            MASK: mask # partial match
            OPTS:
                BATCH_INFO: sbatch path/to/SBATCH.TEMPLATE 10 # Last number is the number of cores
                WFITOPT_GLOBAL: "-hsteps 61 -wsteps 101 -omsteps 81" # Optional, will apply these options to all fits"
                WFITOPTS:
                    - /om_pri/ -ompri 0.31 -dompri 0.01 # At least one option is required. The name in the /'s is a human readable label
                    - /cmb_pri/ -cmb_sim -sigma_Rcmb 0.007 # Optionally include as many other fitopts as you want.

```

### Analyse

The final step in the Pippin pipeline is the Analyse task. It creates a final output directory, moves relevant files into it,
and generates extra plots. It will save out compressed CosmoMC chains and the plotting scripts (so you can download
the entire directory and customise it without worrying about pointing to external files), it will copy in Hubble diagrams,
and - depending on if you've told it to, will make histogram comparison plots between data and sim. Oh and also
redshift evolution plots. The scripts which copy/compress/rename external files into the analyse directory are generally
named `parse_*.py`. So `parse_cosmomc.py` is the script which finds, reads and compresses the MCMC chains from CosmoMC into
the output directory. Then `plot_cosmomc.py` reads those compressed files to make the plots. 

Cosmology contours will be blinded when made by looking at the BLIND flag set on the data. For data, this defaults to
True.

Note that all the plotting scripts work the same way - `Analyse` generates a small yaml file pointing to all the 
resources called `input.yml`, and each script uses the same file to make different plots. It is thus super easy to add your own 
plotting code scripts, and you can specify arbitrary code to execute using the `ADDITIONAL_SCRIPTS` keyword in opts.
Just make sure your code takes `input.yml` as an argument. As an example, to rerun the CosmoMC plots, you'd simply have to 
run `python plot_cosmomc.py input.yml`.

```yaml
ANALYSE:
  SOMELABEL:
    MASK_COSMOFIT: mask  # partial match
    MASK_BIASCOR: mask # partial match
    MASK_LCFIT: [D_DESSIM, D_DATADES] # Creates histograms and efficiency based off the input LCFIT_SIMNAME matches. Optional
    OPTS:
      COVOPTS: [ALL, NOSYS] # Optional. Covopts to match when making contours. Single or list. Exact match.
      SHIFT: False  # Defualt False. Shift all the contours on top of each other
      PRIOR: 0.01  # Default to None. Optional normal prior around Om=0.3 to apply for sims if wanted.
      ADDITIONAL_SCRIPTS: /somepath/to/your/script.py  # Should take the input.yml as an argument
```
