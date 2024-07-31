# 3. CLASSIFICATION

Within Pippin, there are many different classifiers implemented. Most classifiers need to be trained, and can then run in predict mode. All classifiers that require training can either be trained in the same yml file, or you can point to an external serialised instance of the trained class and use that. The general syntax for a classifier is:

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

## SCONE Classifier

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

## SuperNNova Classifier

The [SuperNNova classifier](https://github.com/supernnova/SuperNNova) is a recurrent neural network that operates on simulation photometry. It has three in vuilt variants - its normal (vanilla) mode, a Bayesian mode and a Variational mode. After training, a `model.pt` can be found in the output directory, which you can point to from a different yaml file. You can define a classifier like so:

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

Pippin also allows for SuperNNova input yaml files to be passed, instead of having to define all of the options in the Pippin input yaml. This is done via:

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

## SNIRF Classifier

The [SNIRF classifier](https://github.com/evevkovacs/ML-SN-Classifier) is a random forest running off SALT2 summary statistics. You can specify which features it gets to train on, which has a large impact on performance. After training, there should be a `model.pkl` in the output directory. You can specify one like so:

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

Similar to SNIRF, NN trains on SALT2 summary statistics using a basic Nearest Neighbour algorithm from sklearn. It will produce a `model.pkl` file in its output directory when trained. You can configure it as per SNIRF:

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

Sometimes you want to cheat, and if you have simulations, this is easy. The perfect classifier looks into the sims to get the actual type, and will then assign probabilities as per your configuration. This classifier has no training mode, only predict.

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

To emulate a spectroscopically confirmed sample, or just to save time, we can assign every event a probability of 1.0 that it is a type Ia. As it just returns 1.0 for everything, it only has a predict mode

```yaml
CLASSIFICATION:
  UNITY:
    CLASSIFIER: UnityClassifier
    MODE: predict
```

#### FitProb Classifier

Another useful debug test is to just take the SALT2 fit probability calculated from the chi2 fitting and use that as our probability. You'd hope that classifiers all improve on this. Again, this classifier only has a predict mode.

```yaml
CLASSIFICATION:
  FITPROBTEST:
    CLASSIFIER: FitProbClassifier
    MODE: predict
```
