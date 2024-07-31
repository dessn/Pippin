# 2. LCFIT

This task runs the SALT2 light curve fitting process on light curves from the simulation or DataPrep task. As with the <project:./sim.md> stage, if something goes wrong, Pippin will attempt to give a good reason why. The task is specified like so:

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
