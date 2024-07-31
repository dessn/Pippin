# 9. Analyse

The final step in the Pippin pipeline is the Analyse task. It creates a final output directory, moves relevant files into it, and generates extra plots. It will save out compressed CosmoMC chains and the plotting scripts (so you can download the entire directory and customise it without worrying about pointing to external files), it will copy in Hubble diagrams, and - depending on if you've told it to, will make histogram comparison plots between data and sim. Oh and also redshift evolution plots. The scripts which copy/compress/rename external files into the analyse directory are generally named `parse_*.py`. So `parse_cosmomc.py` is the script which finds, reads and compresses the MCMC chains from CosmoMC into the output directory. Then `plot_cosmomc.py` reads those compressed files to make the plots. 

Cosmology contours will be blinded when made by looking at the BLIND flag set on the data. For data, this defaults to True.

Note that all the plotting scripts work the same way - `Analyse` generates a small yaml file pointing to all the  resources called `input.yml`, and each script uses the same file to make different plots. It is thus super easy to add your own plotting code scripts, and you can specify arbitrary code to execute using the `ADDITIONAL_SCRIPTS` keyword in opts. Just make sure your code takes `input.yml` as an argument. As an example, to rerun the CosmoMC plots, you'd simply have to run `python plot_cosmomc.py input.yml`.

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
