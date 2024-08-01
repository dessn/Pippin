# Using Pippin

```{figure} ../_static/images/console.gif
:alt: Console Output

The console output from a succesfull Pippin run. Follow these instructions and you too can witness a beautiful wall of green text!
```

Using Pippin is very simple. In the top level directory, there is a `pippin.sh`. If you're on Midway and use SNANA, this script will be in your path already. To use Pippin, all you need is a config file, examples of which can be found in the [configs directory](https://github.com/dessn/Pippin/tree/main/configs). Given the config file `example.yml`, simply run `pippin.sh example.yml` to invoke Pippin. This will create a new folder in the `OUTPUT: output_dir` path defined in the global [cfg.yml](https://github.com/dessn/Pippin/blob/main/cfg.yml) file. By default, this is set to the `$PIPPIN_OUTPUT` environment variable, so please either set said variable or change the associated line in the `cfg.yml`.

<details>
  <summary>For the morbidly curious, here's a small demo video of using Pippin in the Midway environment</summary>

```{eval-rst}
.. youtube:: pCaPvzFCZ-Y
    :width: 100%
    :align: center
```

</details>

## Creating your own configuration file

Each configuration file is represented by a yaml dictionary linking each stage to a dictionary of tasks, the key being the unique name for the task and the value being its specific task configuration.

For example, to define a configuration with two simulations and one light curve fitting task (resulting in 2 output simulations and 2 output light curve tasks - one for each simulation), a user would define:

```yaml
SIM:
  SIM_NAME_1:
    SIM_CONFIG: HERE
  SIM_NAME_2:
    SIM_CONFIG: HERE
    
LCFIT:
  LCFIT_NAME_1:
    LCFIT_CONFIG: HERE
```

Configuration detail for each tasks can be found in the <project:./tasks.md> section, with stage-specific example config files available in the [examples directory](https://github.com/dessn/Pippin/tree/main/examples)

## What If I change my config file?

Happens all the time, don't even worry about it. Just start Pippin again and run the file again. Pippin will detect
any changes in your configuration by hashing all the input files to a specific task. So this means, even if you're 
config file itself doesn't change, changes to an input file it references (for example, the default DES simulation
input file) would result in Pippin rerunning that task. If it cannot detect anything has changed, and if the task
finished successfully the last time it was run, the task is not re-executed. You can force re-execution of tasks using the `-r` flag.

## Command Line Arguments

On top of this, Pippin has a few command line arguments, which you can detail with `pippin.sh -h`, but I'll also detail here:

```
  -h                 Show the help menu
  -v, --verbose      Verbose. Shows debug output. I normally have this option enabled.
  -r, --refresh      Refresh/redo - Rerun tasks that completed in a previous run even if the inputs haven't changed.
  -c, --check        Check that the input config is valid but don't actually run any tasks.
  -s, --start        Start at this task and refresh everything after it. Number of string accepted
  -f, --finish       Finish at this stage. For example -f 3 or -f CLASSIFY to run up to and including classification. 
  -p, --permission   Fix permissions and groups on all output, don't rerun
  -i, --ignore       Do NOT regenerate/run tasks up to and including this stage.
  -S, --syntax       If no task is given, prints out the possible tasks. If a task name or number is given, prints the docs on that task. For instance 'pippin.sh -S 0' and 'pippin.sh -S DATAPREP' will print the documentation for the DATAPREP task.
```

For an example, to have a verbose output configuration run and only do data preparation and simulation, 
you would run

`pippin.sh -v -f 1 configfile.yml`


## Stages in Pippin

You may have noticed above that each stage has a numeric idea for convenience and lexigraphical sorting.

The current stages are:

- <project:./tasks/dataprep.md>: Data preparation
- <project:./tasks/sim.md>: Simulation
- <project:./tasks/lcfit.md>: Light curve fitting
- <project:./tasks/classify.md>: Classification (training and testing)
- <project:./tasks/agg.md>: Aggregation (comparing classifiers)
- <project:./tasks/merge.md>: Merging (combining classifier and FITRES output)
- <project:./tasks/biascor.md>: Bias corrections using BBC
- <project:./tasks/createcov.md>: Determine the systematic covariance matrix
- <project:./tasks/cosmofit.md>: Fit hubble diagram and produce cosmological constraint
- <project:./tasks/analyse.md>: Create final output and plots. 

## Pippin on Midway

On midway, sourcing the SNANA setup will add environment variables and Pippin to your path.

Pippin itself can be found at `$PIPPIN`, output at `$PIPPIN_OUTPUT` (which goes to a scratch directory), and `pippin.sh` will automatically work from
any location.

Note that you only have 100 GB on scratch. If you fill that up and need to nuke some files, look both in `$SCRATCH_SIMDIR` to remove SNANA 
photometry and `$PIPPIN_OUTPUT` to remove Pippin's output. I'd recommend adding this to your `~/.bashrc` file to scan through directories you own and 
calculate directory size so you know what's taking the most space. After adding this and sourcing it, just put `dirusage` into the terminal
in both of those locations and see what's eating your quota.

```sh
function dirusage {
    for file in $(ls -l | grep $USER | awk '{print $NF}')
    do
        du -sh "$file"
    done
}
```

## Pippin on Perlmutter

On perlmutter, add `source /global/cfs/cdirs/lsst/groups/TD/setup_td.sh` to your `~/.bashrc` to load all the relevant paths and environment variables.

This will add the `$PIPPIN_DIR` path for Pippin source code, and `$PIPPIN_OUTPUT` for the output of Pippin jobs. Additionally `pippin.sh` can be run from any directory.

To load the perlmutter specific `cfg.yml` you must add the following to the start of your Pippin job:
```yaml
GLOBAL:
    CFG_PATH: $SNANA_LSST_ROOT/starterKits/pippin/cfg_lsst_perlmutter.yml
```

## Examples

If you want detailed examples of what you can do with Pippin tasks, have a look in the [examples directory](https://github.com/dessn/Pippin/tree/main/examples), pick the task you want to know more about, and have a look over all the options.

Here is a very simple configuration file which runs a simulation, does light curve fitting, and then classifies it using the debug FITPROB classifier.

```yaml
SIM:
  DESSIM:
    IA_G10_DES3YR:
      BASE: surveys/des/sim_ia/sn_ia_salt2_g10_des3yr.input

LCFIT:
  BASEDES:
    BASE: surveys/des/lcfit_nml/des_5yr.nml
  
CLASSIFICATION:
  FITPROBTEST:
    CLASSIFIER: FitProbClassifier
    MODE: predict
```

You can see that unless you specify a `MASK` on each subsequent task, Pippin will generally try and run everything on everything. So if you have two simulations defined, you don't need two light curve fitting tasks, Pippin will make one light curve fit task for each simulation, and then two classification tasks, one for each light curve fit task.

## Best Practice

Here are a few best practices for improving your chance of success with Pippin.

### Use `screen`

Pippin jobs can take a long time, so to avoid having to keep a terminal open and an ssh session active for the length of the entire run, it is *highly recommended* you run Pippin in a `screen` session.

For example, if you are doing machine-learning testing, you may create a new screen session called `ml` by running `screen -S ml`. It will then launch a new instance of bash for you to play around in. Conda will **not work out of the box**. To make it work again, run `conda deactivate` and then `conda activate`, and you can check this works by running `which python` and verifying its pointing to the miniconda install. You can then run Pippin as per normal: `pippin.sh -v your_job.yml` and get the coloured output. To leave the screen session, but **still keep Pippin running even after you log out**, press `Ctrl-A`, `Ctrl-D`. As in one, and then the other, not `Ctrl-A-D`. This will detach from your screen session but keep it running. Just going `Ctrl_D` will disconnect and shut it down. To get back into your screen session, simply run `screen -r ml` to reattach. You can see your screen sessions using `screen -ls`.

You may notice if you log in and out of midway that your screen sessions might not show up. This is because midway has multiple head nodes, and your screen session exists only on one of them. This is why when I ssh to midway I specify a specific login node instead of being assigned one. To make it simpler, I'd recommend setting your ssh host in your `.ssh/config` to something along the lines of: 

```sh
Host midway2
    HostName midway2-login1.rcc.uchicago.edu
    User username
```

### Make the most of command line options

There are a number of command line options that are particularly useful. Foremost amongst them is `-v, --verbose` which shows debug output when running Pippin. Including this flag in your run makes it significantly easier to diagnose if anything goes wrong.

The next time saving flag is `-c, --check`, which will do an initial passthrough of your input yaml file, pointing out any obvious errors before anything runs. This is particularly useful if you have long jobs and want to catch bugs early.

The final set of useful flags are the `-s, --start`, `-f, --finish`, and `-i, --ignore`. These allow you to customize exactly what parts of your full job Pippin runs. Pippin decides whether or not it should rerun a task based on a hash generated each time it's run. This hash produced based on the input, these flags are particularly useful if you change your input but *don't want stages to rerun*, such as if you are making small changes to a final stage, or debugging an early stage.

## Advanced Usage

The following are a number of advanced features which aren't required to use Pippin but can drastically improve your experience with Pippin.

### Yaml Anchors

If you are finding that your config files contain lots of duplicated sections (for example, many simulations configured almost the same way, but with one difference), consider using yaml anchors. A thorough explanation of how to use them is available [here](https://blog.daemonl.com/2016/02/yaml.html), however the basics are as follows. First you should add a new taml section at the tope of your input file. The name of this section doesn't matter as long as it doesn't clash with other Pippin stages, however I usually use `ALIAS`. Within this section, you include all of the yaml anchors you need. An example is shown below:

```yaml
ALIAS:
    LOWZSIM_IA: &LOWZSIM_IA
        BASE: surveys/lowz/sims_ia/sn_ia_salt2_g10_lowz.input

SIM:
    SIM_1:
        IA_G10_LOWZ:
            <<: *LOWZSIM_IA
            # Other options here
    SIM_2:
        IA_G10_LOWZ:
            <<: *LOWZSIM_IA
            # Different options here
```

### Include external aliases

**This is new and experimental, use with caution**.

*Note that this is* **not** *yaml compliant*.

When dealing with especially large jobs, or suites of jobs you might find yourself having very large `ALIAS`/`ANCHOR` blocks which are repated amongst a number of Pippin jobs. A cleaner alternative is to have a number of `.yml` files containing your anchors, and then `including` these in your input files which will run Pippin jobs. This way you can share anchors amongst multiple Pippin input files and update them all at the same time. In order to achieve this, Pippin can *preprocess* the input file to directly copy the anchor file into the job file. An example is provided below:

`base_job_file.yml`
```yaml
# Values surround by % indicate preprocessing steps.
# The preprocess below will copy the provided yml files into this one before this one is read in, allowing anchors to propegate into this file
# They will be copied in, in the order you specify, with duplicate tasks merging.
# Note that whitespace before or after the % is fine, as long as % is the first and last character.

# % include: path/to/anchors_sim.yml %
# %include: path/to/anchors_lcfit.yml%

SIM:
  DESSIM:
    IA_G10_DES3YR:
      BASE: surveys/des/sims_ia/sn_ia_salt2_g10_des3yr.input
    GLOBAL:
      # Note that this anchor doesn't exist in this file
      <<: *SIM_GLOBAL
  LCSIM:
    IA_G10_LOWZ:
      BASE: surveys/lowz/sims_ia/sn_ia_salt2_g10_lowz.input
    GLOBAL:
      # Note that this anchor doesn't exist in this file
      <<: *SIM_GLOBAL

LCFIT:
  LS:
    BASE: surveys/lowz/lcfit_nml/lowz.nml
    MASK: DATALOWZ
    FITOPTS: surveys/lowz/lcfit_fitopts/lowz.yml
    # Note that this anchor doesn't exist in this file
    <<: *LCFIT_OPTS
    
  DS:
    BASE: surveys/des/lcfit_nml/des_3yr.nml
    MASK: DATADES
    FITOPTS: surveys/des/lcfit_fitopts/des.yml
    # Note that this anchor doesn't exist in this file
    <<: *LCFIT_OPTS
```

`anchors_sim.yml`
```yaml
ANCHORS_SIM:
    SIM_GLOBAL: &SIM_GLOBAL
        W0_LAMBDA: -1.0
        OMEGA_MATTER: 0.3
        NGEN_UNIT: 0.1
```

`anchors_lcfit.yml`
```yaml
ANCHORS_LCFIT:
    LCFIT_OPTS: &LCFIT_OPTS
        SNLCINP:
            USE_MINOS: F
```

This will be preprocessed to produce the following yaml file, which pippin will then run on.

`final_pippin_input.yml`
```yaml
# Original input file: path/to/base_job_file.yml
# Values surround by % indicate preprocessing steps.
# The preprocess below will copy the provided yml files into this one before this one is read in, allowing anchors to propegate into this file
# They will be copied in, in the order you specify, with duplicate tasks merging.
# Note that whitespace before or after the % is fine, as long as % is the first and last character.

# Anchors included from path/to/anchors_sim.yml
ANCHORS_SIM:
    SIM_GLOBAL: &SIM_GLOBAL
        W0_LAMBDA: -1.0
        OMEGA_MATTER: 0.3
        NGEN_UNIT: 0.1

# Anchors included from path/to/anchors_lcfit.yml
ANCHORS_LCFIT:
    LCFIT_OPTS: &LCFIT_OPTS
        SNLCINP:
            USE_MINOS: F
  
SIM:
  DESSIM:
    IA_G10_DES3YR:
      BASE: surveys/des/sims_ia/sn_ia_salt2_g10_des3yr.input
    GLOBAL:
      <<: *SIM_GLOBAL
  LCSIM:
    IA_G10_LOWZ:
      BASE: surveys/lowz/sims_ia/sn_ia_salt2_g10_lowz.input
    GLOBAL:
      <<: *SIM_GLOBAL

LCFIT:
  LS:
    BASE: surveys/lowz/lcfit_nml/lowz.nml
    MASK: DATALOWZ
    FITOPTS: surveys/lowz/lcfit_fitopts/lowz.yml
    <<: *LCFIT_OPTS
    
  DS:
    BASE: surveys/des/lcfit_nml/des_3yr.nml
    MASK: DATADES
    FITOPTS: surveys/des/lcfit_fitopts/des.yml
    <<: *LCFIT_OPTS
```

Now you can include the `anchors_sim.yml` and `anchors_lcfit.yml` anchors in any pippin job you want, and need only update those anchors once. There are a few caveats to this to be aware of. The preprocessing does not checking to ensure the given file is valid yaml, it simply copies the yaml directly in. As such you should always ensure that the name of your anchor block is unique, any duplicates will mean whichever block is lowest will overwrite all other blocks of the same name. Additionally, whilst you could technically use this to store Pippin task blocks in external yml files, this is discouraged as this feature was only intended for anchors and aliases.


### Use external results

Often times you will want to reuse the results of one Pippin job in other Pippin jobs, for instance reusing a biascor sim so you don't need to resimulate every time. This can be accomplished via the `EXTERNAL`, `EXTERNAL_DIRS`, and `EXTERNAL_MAP` keywords.

There are in essense two ways of including external tasks. Both operate the same way, one is just a bit more explicit than the other. The explicit way is when adding a task that is an *exact* replica of an external task, you can just add the `EXTERNAL` keyword. For example, in the reference 5YR analysis, all the biascor sims are precomputed, so we can define them as external tasks like this:

```yaml
SIM:
  DESSIMBIAS5YRIA_C11: # A SIM task we don't want to rerun
    EXTERNAL: $PIPPIN_OUTPUT/GLOBAL/1_SIM/DESSIMBIAS5YRIA_C11 # The path to a matching external SIM task, which is already finished
  DESSIMBIAS5YRIA_G10:
    EXTERNAL: $PIPPIN_OUTPUT/GLOBAL/1_SIM/DESSIMBIAS5YRIA_G10
  DESSIMBIAS5YRCC:
    EXTERNAL: $PIPPIN_OUTPUT/GLOBAL/1_SIM/DESSIMBIAS5YRCC
```

In this case, we use the `EXTERNAL` keyword because each of the three defined tasks can only be associated with one, and only one, `EXTERNAL` task. Because `EXTERNAL` tasks are one-to-one with a defined task, the name of the defined task, and the `EXTERNAL` task do not need to match.

Suppose we don't want to recompute the light curve fits. After all, most of the time we're not changing that step anyway! However, unlike `SIM`, `LCFIT` runs multiple sub-tasks - one for each `SIM` task you are performing lightcurve fitting on.

```yaml
LCFIT:
  D: # An LCFIT task we don't want to rerun
    BASE: surveys/des/lcfit_nml/des_5yr.nml
    MASK: DESSIM # Selects a subset of SIM tasks to run lightcurve fitting on
                 # In this case, the SIM tasks are DESSIMBIAS5YRIA_C11, DESSIMBIAS5YRIA_G10, and DESSIMBIAS5YRCC
    EXTERNAL_DIRS:
      - $PIPPIN_OUTPUT/GLOBAL/2_LCFIT/D_DESSIMBIAS5YRIA_C11 # Path to a previously run LCFIT sub-task
      - $PIPPIN_OUTPUT/GLOBAL/2_LCFIT/D_DESSIMBIAS5YRIA_G10
      - $PIPPIN_OUTPUT/GLOBAL/2_LCFIT/D_DESSIMBIAS5YRCC
```

That is, we have one `LCFIT` task, but because we have three sims going into it and matching the mask, we can't point to a single `EXTERNAL` task. Instead, we provide an external path for each sub-task, as defined in `EXTERNAL_DIRS`. The name of each external sub-task must exactly match the `LCFIT` task name, and the `SIM` sub-task name. For example, the path to the `DESSIMBIAS5YRIA_C11` lightcurve fits, must be `D_DESSIMBIAS5YRIA_C11`.

Note that you still need to point to the right base file, because Pippin still wants those details. It won't be submitted anywhere though, just loaded in. 

To use `EXTERNAL_DIRS` on pre-computed tasks that don't follow your current naming scheme (i.e the `LCFIT` task name, or the `SIM` sub-task names differ), you can make use of `EXTERNAL_MAP` to provide a mapping between the `EXTERNAL_DIR` paths, and each `LCFIT` sub-task.

```yaml
LCFIT:
  D: # An LCFIT task we don't want to rerun
    BASE: surveys/des/lcfit_nml/des_5yer.nml
    MASK: DESSIM # Selects a subset of SIM tasks to run lightcurve fitting on
    EXTERNAL_DIRS: # Paths to external LCFIT tasks, which do not have an exact match with this task
      - $PIPPIN_OUTPUT/EXAMPLE_C11/2_LCFIT/DESFIT_SIM
      - $PIPPIN_OUTPUT/EXAMPLE_G10/2_LCFIT/DESFIT_SIM
      - $PIPPIN_OUTPUT/EXAMPLE/2_LCFIT/DESFIT_CCSIM
    EXTERNAL_MAP:
      # LCFIT_SIM: EXTERNAL_MASK
      D_DESSIMBIAS5YRIA_C11: EXAMPLE_C11 # In this case we are matching to the pippin job name, as the LCFIT task name is shared between two EXTERNAL_DIRS
      D_DESSIMBIAS5YRIA_G10: EXAMPLE_G10 # Same as C11
      D_DESSIMBIAS5YRCC: DESFIT_CCSIM # In this case we match to the LCFIT task name, as the pippin job name (EXAMPLE) would match with the other EXTERNAL_DIRS
```

The flexibility of `EXTERNAL_DIRS` means you can mix both precomputed and non-precomputed tasks together. Take this classificaiton task:

```yaml
CLASSIFICATION:
  SNNTEST:
    CLASSIFIER: SuperNNovaClassifier
    MODE: predict
    OPTS:
      MODEL: $PIPPIN_OUTPUT/GLOBAL/3_CLAS/SNNTRAIN_DESTRAIN/model.pt
    EXTERNAL_DIRS:
      - $PIPPIN_OUTPUT/GLOBAL/3_CLAS/SNNTEST_DESSIMBIAS5YRIA_C11_SNNTRAIN_DESTRAIN
      - $PIPPIN_OUTPUT/GLOBAL/3_CLAS/SNNTEST_DESSIMBIAS5YRIA_G10_SNNTRAIN_DESTRAIN
      - $PIPPIN_OUTPUT/GLOBAL/3_CLAS/SNNTEST_DESSIMBIAS5YRCC_SNNTRAIN_DESTRAIN
```

It will load in the precomputed classification results for the biascor sims, and then also run and generate classification results on any other simulation tasks (such as running on the data) using the pretrained model `model.pt`.

Finally, the way this works under the hood is simple - it copies the directory over explicitly. And it will only copy once, so if you want the "latest version" just ask the task to refresh (or delete the folder). Once it copies it, there is no normal hash checking, it reads in the `config.yml` file created by the task in its initial run and powers onwards.

If you have any issues using this new feature, check out the `ref_des_5yr.yml` file or flick me a message.

### Changing SBATCH options

Pippin has sensible defaults for the sbatch options of each task, however it is possible you may sometimes want to overwrite some keys, or even replace the sbatch template entirely. You can do this via the `BATCH_REPLACE`, and `BATCH_FILE` options respectively.

In order to overwrite the default batch keys, add the following to any task which runs a batch job:

```yaml
BATCH_REPLACE:
    REPLACE_KEY1: value
    REPLACE_KEY2: value 
```

Possible options for `BATCH_REPLACE` are:

* `REPLACE_NAME`: `--job-name`
* `REPLACE_LOGFILE`: `--output`
* `REPLACE_WALLTIME`: `--time`
* `REPLACE_MEM`: `--mem-per-cpu`

Note that changing these could have unforseen consequences, so use at your own risk.

If replacing these keys isn't enough, you are able to create you own sbatch templates and get Pippin to use them. This is useful if you want to change the partition, or add some additional code which runs before the Pippin job. Note that your template **must** contain the keys listed above in order to work properly. In addition you **must** have `REPLACE_JOB` at the bottom of your template file, otherwise Pippin will not be able to load it's jobs into your template. An example template is as follows:

```bash
#!/bin/bash

#SBATCH -p broadwl-lc
#SBATCH --account=pi-rkessler
#SBATCH --job-name=REPLACE_NAME
#SBATCH --output=REPLACE_LOGFILE
#SBATCH --time=REPLACE_WALLTIME
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=REPLACE_MEM
echo $SLURM_JOB_ID starting execution `date` on `hostname`

REPLACE_JOB
```

To have Pippin use your template, simply add the following to your task:

```yaml
BATCH_FILE: path/to/your/batch.TEMPLATE
```

## FAQ

### Pippin is crashing on some task and the error message isn't useful

Feel free to send me the log and stack, and I'll see what I can do turn the exception into something more human-readable.

### I want to modify a ton of files but don't want huge yml files, please help

You can modify input files and put them in a directory you own, and then tell Pippin to look there (in addition to the default location) when its constructing your tasks. To do this, see [this example here](https://github.com/dessn/Pippin/blob/main/examples/global.yml), or use this code snippet at the top of your YAML file (not that it matters if it's at the top):

```yaml
GLOBAL:
  DATA_DIRS:
    - /some/new/directory/with/your/files/in/it
```

### I want to use a different cfg.yml file!

```yaml
GLOBAL:
  CFG_PATH: /your/path/here
```

### Stop rerunning my sims!

For big biascor sims it can be frustrating if you're trying to tweak biascor or later stages and sims kick off
because of some trivial change. So use the `--ignore` ro `-i` command to ignore any undone tasks or tasks with 
hash disagreements in previous stages. To clarify, even tasks that do not have a hash, and have never been submitted, will
not be run if that stage is set to be ignored.  
