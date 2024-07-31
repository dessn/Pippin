# 1. SIM

The simulation task does exactly what you'd think it does. It invokes [SNANA](https://github.com/RickKessler/SNANA) to run some similation as per your configuration. If something goes wrong, Pippin tries to dig through the log files to give you a useful error message, but sometimes this is difficult (i.e. the logs have been zipped up). With the current version of SNANA, each simulation can have at most one Ia component, and an arbitrary number of CC components. The specification for the simulation task config is as follows:

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
