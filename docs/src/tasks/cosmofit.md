# 8. COSMOFIT

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

## CosmoMC

Launching CosmoMC is hopefully fairly simple. There are a list of provided configurations under the `cosmomc_templates` directory (inside `data_files`), and the main job of the user is to pick which one they want. 

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

## WFit

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
