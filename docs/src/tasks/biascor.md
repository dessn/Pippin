# 6. BIASCOR  
- revised May 15 2026 by R.Kessler

Using the FITRES files produce by light curve fitting (2_LCFIT), along with optional classifer probabilities (3_CLASS), this stage runs SALT2mu.exe to implement "BEAMS with Bias Corrections" (BBC). A large SNIa simulation is needed for spec-confirmed and photometrically-identified samples; the latter also needs a large "non-Ia" simulation with Core Collapse SNe and peculiar SNe Ia. Inputs are shown below, along with how to combine surveys (e.g., low-z plus high-z).

```yaml
BIASCOR:
  LABEL:
    # The base input file to utilise
    BASE: surveys/des/bbc/bbc.input
    
    # names of the lcfits_data (real or sim). Note list format, and LcfitLabel_SimLabel key.
    DATA: [DESFIT_DESSIM, LOWZFIT_LOWZSIM]
    
    # Input Ia bias correction simulation per survey
    SIMFILE_BIASCOR: [DESFIT_DESBIASCOR, LOWZFIT_LOWZBIASCOR]

    # For surveys with contamination (e.g., DES-SN5YR), include NONIA-only simulation for CCPRIOR    
    SIMFILE_CCPRIOR: DESFIT_DESSIMBIAS5YRCC

    # Define which classifier PROB-Ia to use by specifying argument of PROB_COLUMN_NAME in 3_CLAS stage.
    CLASSIFIER: PROB_SCONEV19
    
    # - - - - - - - - - - - - rarely used/obsolete features - - - - - - - - - - -
    # Optional, specify FITOPT to use. Defaults to 0 for each SIMFILE_BIASCOR/SIMFILE_CCPRIOR. Must specify a FITOPT for each SIMFILE_BIASCOR/SIMFILE_CCPRIOR
    SIMFILE_BIASCOR_FITOPTS: [0, 1] # FITOPT000 and FITOPT001
    SIMFILE_CCPRIOR_FITOPTS: [0, 1] # FITOPT000 and FITOPT001

    # Default False. If multiple sims (RANSEED_CHANGE), make one or all Hubble plots.
    MAKE_ALL_HUBBLE: False
    
    # Defaults to False. Load recalibrated probabilities.
    USE_RECALIBRATED: True
    # - - - - - - - - - - - end rare/obsolete - - - - - - - - - - - - - - - 

  
  # Optional "MUOPTS" add BBC-related systematics. They share the structure of the main biascor definition. Can define multiple, or use a dict structure, with the MUOPT name being the key
  MUOPTS:
      C11:
        SIMFILE_BIASCOR: [D_DESBIASSYS_C11, L_LOWZBIASSYS_C11]
        SCALE: 0.5 # Default=1, used by CREATE_COV to determine COVSYS contribution
        
  # Generic OPTS that can modify the base file and overwrite properties
  OTPS:
    BATCH_INFO: sbatch $SBATCH_TEMPLATES/SBATCH_Midway2_1hr.TEMPLATE 10

    # optional replace BCOR & CCPRIOR for specific LCFIT-FITOPTS. Initial motivation is for systematics with altered zPHOT algorithm; same zPHOT alteration should be be run for both data and biasCor. Each /path argument can be a comma-sep list (no brackets, no spaces) to include multuple surveys.
    REPLACE_SIMFILE_BIASCOR: 
      label0: /path_replace_bcor0  # label0 must match a FITOPT label at 2_LCFIT stage
      label1: /path_replace_bcor1  # label1 must match a FITOPT label at 2_LCFIT stage
    REPLACE_SIMFILE_CCPRIOR:
      label0: /path_replace_cc0    # label0 must match a FITOPT label at 2_LCFIT stage
      label1: /path_replace_cc1    # label1 must match a FITOPT label at 2_LCFIT stage      
```

For those that generate large simulations and want to cut them up into little pieces, you want the `NSPLITRAN` syntax. The configuration below will take the inputs and divide them into 10 samples.

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
