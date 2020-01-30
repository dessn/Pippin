# Pippin


Pippin - a pipeline for Supernova cosmology analysis

![Pippin Meme](meme.jpg)



## Table of Contents

* [Using Pippin](#using-pippin)
* [Installing Pippin](#installing-it-fresh)
* [Contributing to Pippin](#contributing-to-pippin)
* [Examples](#examples)
* [Adding a new Task](#adding-a-new-task)
* [Adding a new classifier](#adding-a-new-classifier)


## Installing it fresh

If you're using a pre-installed version of Pippin - like the one on Midway, ignore this.

If you're not, installing Pippin is simple.

1. Checkout Pippin
2. Ensure you have the dependencies install (`pip install -r requirements.txt`)
3. Celebrate

Now, Pippin also interfaces with other tasks: SNANA and machine learning classifiers mostly.

I won't cover installing SNANA here, hopefully you already have it. But to install the classifiers, we'll take
[SuperNNova](https://github.com/supernnova/SuperNNova) as an example. To install that, find a good place for it and:

1. Checkout `https://github.com/SuperNNova/SuperNNova`
2. Create a GPU conda env for it: `conda create --name snn_gpu --file env/conda_env_gpu_linux64.txt`
3. Activate environment and install natsort: `conda activate snn_gpu` and `conda install --yes natsort`

Then, in the Pippin config file `cfg.yml` in the top level directory, ensure that the SNN path in Pippin is
pointing to where you just cloned SNN into.

## Using Pippin

Using Pippin is very simple. In the top level directory, there is a `pippin.sh`. If you're on midway and use SNANA, this
script will be on your path already. To use Pippin, all you need is a config file ready to go. I've got a bunch of mine and 
some general ones in the `configs` directory, but you can put yours wherever you want. I recommend adding your initials to the 
front of the file to make it obvious in the shared output directory which folders as yours.

If you have `example.yml` as your config file and want pippin to run it, easy:
`pippin.sh config.yml`


### What If I change my config file?

Happens all the time, don't even worry about it. Just start Pippin again and run the file again. Pippin will detect
any changes in your configuration by hashing all the input files to a specific task. So this means, even if you're 
config file itself doesn't change, changes to an input file it references (for example, the default DES simulation
input file) would result in Pippin rerunning that task. If it cannot detect anything has changed, and if the task
finished successfully the last time it was run, the task is not re-executed. You can force re-execution of tasks using the `-r` flag.


### Command Line Arguments

On top of this, Pippin has a few command line arguments, which you can detail with `pippin.sh -h`, but I'll also detail here:


* `-h`: Show the help menu
* `-v` or `--verbose`: Verbose. Shows debug output. I normally have this option enabled.
* `-r` or `--refresh`: Refresh/redo - Rerun tasks that completed in a previous run even if the inputs haven't changed.
* `-c` or `--check`: Check that the input config is valid but don't actually run any tasks.
* `-s` or `--start`: Start at this task and refresh everything after it. For example, if the underlying SALT2 model has changed (and Pippin wouldn't know about that) and you want to redo all the light curve fitting but not simulating the photometry, you would pass `-s 1` or `-s LCFIT`
* `-f` or `--finish`: Finish at this stage. For example, you may want to go up to classification but not cosmology, you would pass `-f 3` or `-f CLASSIFY`. Not you finish *including* that stage. 

For an example, to have a verbose output configuration run and only do data preparation and simulation, 
you would run

`pippin.sh -vf 1 configfile.yml`


### Stages in Pippin

You may have noticed above that each stage has a numeric idea for convenience and lexigraphical sorting.

The current stages are:

* `0, DATAPREP` Data preparation
* `1, SIM`: Simulation
* `2, LCFIT`: Light curve fitting
* `3, CLASSIFY`: Classification (training and testing)
* `4, AGG`: Aggregation (comparing classifiers)
* `5, MERGE`: Merging (combining classifier and FITRES output)
* `6, BIASCOR`: Bias corrections using BBC
* `7, CREATE_COV`: Create input files needed for CosmoMC
* `8, COSMOMC`: Run CosmoMC and fit cosmology
* `9, ANALYSE`: Create final output and plots. Includes output from CosmoMC, BBC and Light curve fitting.

### Pippin on Midway

On midway, sourcing the SNANA setup will add environment variables and Pippin to your path.

Pippin itself can be found at `$PIPPIN`, output at `$PIPPIN_OUTPUT` (which goes to a scratch directory), and `pippin.sh` will automatically work from
any location.

Note that you only have 100 GB on scratch. If you fill that up and need to nuke some files, look both in `$SCRATCH_SIMDIR` to remove SNANA 
photometry and `$PIPPIN_OUTPUT` to remove Pippin's output. I'd recommend adding this to your `~/.bashrc` file to scan through directories you own and 
calculate directory size so you know what's taking the most space. After adding this and sourcing it, just put `dirusage` into the terminal
in both of those locations and see what's eating your quota.

```bash
function dirusage {
    for file in $(ls -l | grep $USER | awk '{print $NF}')
    do
        du -sh "$file"
    done
}
```

## Contributing to Pippin

Contributing to Pippin is easy. Here are some ways you can do it, in order of preference:

1. Submit an [issue on Github](https://github.com/samreay/Pippin), and then submit a pull request to fix that issue.
2. Submit an [issue on Github](https://github.com/samreay/Pippin), and then wait until I have time to look at it. Hopefully thats quickly, but no guarantees.
3. Email me with a feature request

If you do want to contribute code, fantastic. [Please note that all code in Pippin is subject to the Black formatter](https://black.readthedocs.io/en/stable/). 
I would recommend installing this yourself because it's a great tool.


## Examples

If you want detailed examples of what you can do with Pippin tasks, have a look in the [examples directory](https://github.com/Samreay/Pippin/tree/master/examples),
pick the task you want to know more about, and have a look over all the options.

Here is a very simple configuration file which runs a simulation, does light curve fitting, and then classifies it using the
debug FITPROB classifier.

```yaml
SIM:
  DESSIM:
    IA_G10_DES3YR:
      BASE: sn_ia_salt2_g10_des3yr.input

LCFIT:
  BASEDES:
    BASE: des.nml
  
CLASSIFICATION:
  FITPROBTEST:
    CLASSIFIER: FitProbClassifier
    MODE: predict
```

You can see that unless you specify a `MASK` on each subsequent task, Pippin will generally try and run everything on everything. So if you have two
simulations defined, you don't need two light curve fitting tasks, Pippin will make one light curve fit task for each simulation, and then two classification tasks,
one for each light curve fit task.

### Anchoring in YAML files

If you are finding that your config files contain lots of duplicated sections (for example, many simulations configured
almost the same way but with one differnece), consider using YAML anchors. [See this blog post](https://blog.daemonl.com/2016/02/yaml.html)
for more detail. You can define your anchors in the main config section, or add a new section (like SIM, LCFIT, CLASSIFICATION). So long as it doesn't
match a Pippin keyword for each stage, you'll be fine. I recommend `ANCHORS:` or `GLOBAL:` or `DEFAULTS:` at the top of the file, all of those will work.


*********

**Warning, developer doco below**

*********


## Adding a new task

Alright there, you want to add a new task to Pippin? Great. Here's what you've got to do:

1. Create an implementation of the `Task` class, can keep it empty for now.
2. Figure out where it goes - in `manager.py` at the top you can see the current stages in Pippin. You'll probably need to figure out where it should go. 
Once you have figured it out, import the task and slot it in.
3. Back in your new class that extends Task, you'll notice you have a few methods to implement:
    1. `_run(force_refresh)`: Kick the task off, report True or False for successful kicking off. Determine if you need to rerun the task using both a hash and `force_refresh`. 
    To help with determining the hash, there are a few hand functions: `get_hash_from_string`, `save_hash`, `get_hash_from_files`, `get_old_hash`. See, for example, the Analyse 
    task for an example on how I use these.
    2. `_check_completion(squeue)`: Check to see if the task (whether its being rerun or not) is done. 
    Normally I do this by checking for a done file, which contains either SUCCESS or FAILURE. For example, if submitting a script to a queuing system, I might have this after the primary command:
        ```batch
        if [ $? -eq 0 ]; then
            echo SUCCESS > {done_file}
        else
            echo FAILURE > {done_file}
        fi
        ```
        This allows me to easily see if a job failed or passed. On failure, I then generally recommend looking through the task logs and trying to figure out what went wrong, so you can present a useful message
        to your user. 
        To then show that error, or **ANY MESSAGE TO THE USER**, use the provided logger:
        `self.logger.error("The task failed because of this reason")`. 
        
        This method should return either a) Task.FINISHED_FAILURE, Task.FINISHED_SUCCESS, or alternatively the number of jobs still in the queue, which you could figure out because I pass in all jobs the user has
        active in the variable squeue (which can sometimes be None).
    3. `get_tasks(task_config, prior_tasks, output_dir, stage_num, prefix, global_config)`: From the given inputs, determine what tasks should be created, and create them, and then return them in a list. For context,
    here is the code I use to determine what simulation tasks to create:
        ```python
        @staticmethod
        def get_tasks(config, prior_tasks, base_output_dir, stage_number, prefix, global_config):
            tasks = []
            for sim_name in config.get("SIM", []):
                sim_output_dir = f"{base_output_dir}/{stage_number}_SIM/{sim_name}"
                s = SNANASimulation(sim_name, sim_output_dir, f"{prefix}_{sim_name}", config["SIM"][sim_name], global_config)
                Task.logger.debug(f"Creating simulation task {sim_name} with {s.num_jobs} jobs, output to {sim_output_dir}")
                tasks.append(s)
            return tasks
        ```

## Adding a new classifier

Alright, so what if we're not after a brand new task, but just adding another classifier. Well, its easier to do, and I recommend looking at 
`nearest_neighbor_python.py` for something to copy from. You'll see we have the parent Classifier class, I write out the slurm script that
would be used, and then define the `train` and `predict` method (which both invoke a general `classify` function in different ways, you can do this
however you want.)

You'll also notice a very simply `_check_completion` method, and a `get_requirmenets` method. The latter returns a two-tuple of booleans, indicating 
whether the classifier needs photometry and light curve fitting results respectively. For the NearestNeighbour code, it classifies based
only on SALT2 features, so I return `(False, True)`.

Finally, you'll need to add your classifier into the ClassifierFactory in `classifiers/factory.py`, so that I can link a class name
in the YAML configuration to your actual class. Yeah yeah, I could use reflection or dynamic module scanning or similar, but I've had issues getting
the behaviour consistent across systems and conda environments, so we're doing it the hard way.