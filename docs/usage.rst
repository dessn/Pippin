############
Using Pippin
############

Using Pippin is very simple. In the top level directory, there is a ``pippin.sh`` script. If you're on midway and use SNANA, this script will be on your path already. Otherwise you can add it to your path by adding the following to your ``.bashrc``:

.. code-block:: sh

    export PATH=$PATH:"path/to/pippin"

To use Pippin, all you need is a config file ready to go. I've got a bunch of mine and some general ones in the configs directory, but you can put yours wherever you want. I recommend adding your initials to the front of the file to make it obvious in the shared output directory which folders as yours.

If you have ``example.yml`` as your config file and want pippin to run it, simply run ``pippin.sh example.yml``.

The file name that you pass in should contain a run configuration. Note that this is different to the global software configuration file ``cfg.yml``, and remember to ensure that your ``cfg.yml`` file is set up properly and that you know where you want your output to be installed. By default, I assume that the ``$PIPPIN_OUTPUT`` environment variable is set as the output location, so please either set said variable or change the associated line in the cfg.yml. For the morbidly curious, `here <https://www.youtube.com/watch?v=pCaPvzFCZ-Y>`__ is a very small demo video of using Pippin in the Midway environment.

.. image:: _static/images/console.gif

Creating your own configuration file
=====================================

Each configuration file is represented by a yaml dictionary linking each stage (see stage declaration section below) to a dictionary of tasks, the key being the unique name for the task and the value being its specific task configuration.

For example, to define a configuration with two simulations and one light curve fitting task (resulting in 2 output simulations and 2 output light curve tasks - one for each simulation), a user would define:

.. code-block:: yaml

    SIM:
        SIM_NAME_1:
            SIM_CONFIG: HERE
        SIM_NAME_2:
            SIM_CONFIG: HERE

    LCFIT:
        LCFIT_NAME_1:
            LCFIT_CONFIG: HERE

The available tasks and their configuration details can be found in the :doc:`Tasks <tasks>` section. Alternatively, you can see examples in the ``examples`` directory for each task.

Command Line Arguments
=======================

Pippin has a number of useful command line arguments which you can quickly reference via ``pippin.sh -h``.

.. code-block:: sh

    -h, --help            show this help message and exit
    --config CONFIG       Location of global config (i.e. cfg.yml)
    -v, --verbose         increase output verbosity
    -s START, --start START
                          Stage to start and force refresh. Accepts either the
                          stage number or name (i.e. 1 or SIM)
    -f FINISH, --finish FINISH
                          Stage to finish at (it runs this stage too). Accepts
                          either the stage number or name (i.e. 1 or SIM)
    -r, --refresh         Refresh all tasks, do not use hash
    -c, --check           Check if config is valid
    -p, --permission      Fix permissions and groups on all output, don't rerun
    -i IGNORE, --ignore IGNORE
                          Dont rerun tasks with this stage or less. Accepts
                          either the stage number of name (i.e. 1 or SIM)
    -S [SYNTAX], --syntax [SYNTAX]
                          Get the syntax of the given stage. Accepts either the
                          stage number or name (i.e. 1 or SIM). If run without
                          argument, will tell you all stage numbers / names.
    
As an example, to have a verbose output configuration run and only do data preperation and simulation, you would run ``pippin.sh -vf 1 configfile.yml``.

Pippin on Midway
=================

On midway, sourcing the SNANA setup will add environment variables and Pippin to your path.

Pippin itself can be found at ``$PIPPIN``, output at ``$PIPPIN_OUTPUT`` (which goes to a scratch directory), and ``pippin.sh`` will automatically work from any location.

Note that you only have 100 GB on scratch. If you fill that up and need to nuke some files, look both in ``$SCRATCH_SIMDIR`` to remove SNANA photometry and ``$PIPPIN_OUTPUT`` to remove Pippin's output. Running the ``dirusage`` command on midway will (after some time) give you a list of which directories are taking up the most space.

Examples
========

If you want detailed examples of what you can do with Pippin tasks, have a look in the `examples directory <https://github.com/dessn/Pippin/tree/main/examples>`__, pick the task you want to know more about, and have a look over all the options.

Here is a very simple configuration file which runs a simulation, does light curve fitting, and then classifies it useing the debug FITPROB classifier.

.. code-block:: yaml

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

You can see that unless you specify a ``MASK`` on each subsequent task, Pippin will generally try and run everything on everything. So if you have two simulations defined, you don't need two light curve fitting tasks, Pippin will make one light curve fit task for each simulation, and then two classification tasks, one for each light curve fit task.

Best Practice
==============

Here are a few best practices for improving your chance of success with Pippin.

Use ``screen``
---------------

Pippin jobs can take a long time, so to avoid having to keep a terminal open and an ssh session active for the length of the entire run, it is *highly recommended* you run Pippin in a ``screen`` session.

For example, if you are doing machine-learning testing, you may create a new screen session called ml by running ``screen -S ml``. It will then launch a new instance of bash for you to play around in. conda will **not work out of the box**. To make it work again, run ``conda deactivate`` and then ``conda activate``, and you can check this works by running ``which python`` and verifying its pointing to the miniconda install. You can then run Pippin as per normal: ``pippin.sh -v your_job.yml`` and get the coloured output. To leave the screen session, but **still keep Pippin running even after you log out**, press ``Ctrl-A``, ``Ctrl-D``. As in one, and then the other, not ``Ctrl-A-D``. This will detach from your screen session but keep it running. Just going ``Ctrl_D`` will disconnect and shut it down. To get back into your screen session, simply run ``screen -r ml`` to reattach. You can see your screen sessions using ``screen -ls``.

You may notice if you log in and out of midway that your screen sessions might not show up. This is because midway has multiple head nodes, and your screen session exists only on one of them. This is why when I ssh to midway I specify a specific login node instead of being assigned one. To make it simpler, I'd recommend setting your ssh host in your ``.ssh/config`` to something along the lines of: 

.. code-block:: sh

    Host midway2
        HostName midway2-login1.rcc.uchicago.edu
        User username

Make the most of command line options
---------------------------------------

There are a number of command line options that are particularly useful. Foremost amongst them is ``-v, --verbose`` which shows debug output when running Pippin. Including this flag in your run makes it significantly easier to diagnose if anything goes wrong.

The next time saving flag is ``-c, --check``, which will do an initial passthrough of your input yaml file, pointing out any obvious errors before anything runs. This is particularly useful if you have long jobs and want to catch bugs early.

The final set of useful flags are the ``-s, --start``, ``-f, --finish``, and ``-i, --ignore``. These allow you to customize exactly what parts of your full job Pippin runs. Pippin decides whether or not it should rerun a task based on a hash generated each time it's run. This hash produced based on the input, these flags are particularly useful if you change your input but *don't want stages to rerun*, such as if you are making small changes to a final stage, or debugging an early stage.

Advanced Usage
==============

The following are a number of advanced features which aren't required to use Pippin but can drastically improve your experience with Pippin.

Yaml Anchors
-------------

If you are finding that your config files contain lots of duplicated sections (for example, many simulations configured almost the same way, but with one difference), consider using yaml anchors. A thorough explanation of how to use them is available `here <https://blog.daemonl.com/2016/02/yaml.html>`__, however the basics are as follows. First you should add a new taml section at the tope of your input file. The name of this section doesn't matter as long as it doesn't clash with other Pippin stages, however I usually use `ALIAS`. Within this section, you include all of the yaml anchors you need. An example is shown below:

.. code-block:: yaml

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

Include external aliases
------------------------
**This is new and experimental, use with caution**.

*Note that this is* **not** *yaml compliant*.

When dealing with especially large jobs, or suites of jobs you might find yourself having very large ``ALIAS``/``ANCHOR`` blocks which are repated amongst a number of Pippin jobs. A cleaner alternative is to have a number of ``.yml`` files containing your anchors, and then ``including`` these in your input files which will run Pippin jobs. This way you can share anchors amongst multiple Pippin input files and update them all at the same time. In order to achieve this, Pippin can *preprocess* the input file to directly copy the anchor file into the job file. An example is provided below:

``base_job_file.yml``

.. code-block:: yaml

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

``anchors_sim.yml``

.. code-block:: yaml

    ANCHORS_SIM:
        SIM_GLOBAL: &SIM_GLOBAL
            W0_LAMBDA: -1.0
            OMEGA_MATTER: 0.3
            NGEN_UNIT: 0.1

``anchors_lcfit.yml``

.. code-block:: yaml

    ANCHORS_LCFIT:
        LCFIT_OPTS: &LCFIT_OPTS
            SNLCINP:
                USE_MINOS: F

This will be preprocessed to produce the following yaml file, which pippin will then run on.

``final_pippin_input.yml``

.. code-block:: yaml

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

Now you can include the ``anchors_sim.yml`` and ``anchors_lcfit.yml`` anchors in any pippin job you want, and need only update those anchors once. There are a few caveats to this to be aware of. The preprocessing does not checking to ensure the given file is valid yaml, it simply copies the yaml directly in. As such you should always ensure that the name of your anchor block is unique, any duplicates will mean whichever block is lowest will overwrite all other blocks of the same name. Additionally, whilst you could technically use this to store Pippin task blocks in external yml files, this is discouraged as this feature was only intended for anchors and aliases.


Use external results
---------------------

Often times you will want to reuse the results of one Pippin job in other Pippin jobs, for instance reusing a biascor sim so you don't need to resimulate every time. This can be accomplished via the ``EXTERNAL`` and ``EXTERNAL_DIR`` keywords.

The ``EXTERNAL`` keyword is used when you only need to specify a single external result, such as when you are loading in a simulation. If that's the case you simply need to let Pippin know where the external results are located. An example loading in external biascor sims is below:

.. code-block:: yaml

    SIM:
        DESSIMBIAS5YRIA_C11:
            EXTERNAL: $PIPPIN_OUTPUT/GLOBAL/1_SIM/DESSIMBIAS5YRIA_C11
        DESSIMBIAS5YRIA_G10:
            EXTERNAL: $PIPPIN_OUTPUT/GLOBAL/1_SIM/DESSIMBIAS5YRIA_G10
        DESSIMBIAS5YRCC:
            EXTERNAL: $PIPPIN_OUTPUT/GLOBAL/1_SIM/DESSIMBIAS5YRCC

The ``EXTERNAL_DIRS`` keyword is used when there isn't a one-to-one mapping between the task the external results. An example of this is a lightcurve fitting task where a single task will fit multiple lightcurves. If this is the case, you can specify a number of external results using the ``EXTERNAL_DIRS`` keyword:

.. code-block:: yaml

    LCFIT:
        D:
            BASE: surveys/des/lcfit_nml/des_5yr.nml
            MASK: DESSIM
            EXTERNAL_DIRS:
                - $PIPPIN_OUTPUT/GLOBAL/2_LCFIT/D_DESSIMBIAS5YRIA_C11
                - $PIPPIN_OUTPUT/GLOBAL/2_LCFIT/D_DESSIMBIAS5YRIA_G10
                - $PIPPIN_OUTPUT/GLOBAL/2_LCFIT/D_DESSIMBIAS5YRCC

Note that in this case the name of the external results matches the name of the task. Any tasks which do not have an exact match in ``EXTERNAL_DIRS`` are run as normal, allowing you to mix and match both precomputed and non-precomputed tasks together.

If you have external results which don't have an exact match but should still be used, you can specify how the external results should be used via the ``EXTERNAL_MAP`` keyword:

.. code-block:: yaml

    LCFIT:
        D:
            BASE: surveys/des/lcfit_nml/des_5yer.nml
            MASK: DESSIM
            EXTERNAL_DIRS:
                - $PIPPIN_OUTPUT/EXAMPLE_C11/2_LCFIT/DESFIT_SIM
                - $PIPPIN_OUTPUT/EXAMPLE_G10/2_LCFIT/DESFIT_SIM
                - $PIPPIN_OUTPUT/EXAMPLE/2_LCFIT/DESFIT_CCSIM
            EXTERNAL_MAP:
                # LCFIT_SIM: EXTERNAL_MASK
                D_DESSIMBIAS5YRIA_C11: EXAMPLE_C11 # In this case we are matching to the pippin job name, as the LCFIT task name is shared between two EXTERNAL_DIRS
                D_DESSIMBIAS5YRIA_G10: EXAMPLE_G10 # Same as C11
                D_DESSIMBIAS5YRCC: DESFIT_CCSIM # In this case we match to the LCFIT task name, as the pippin job name (EXAMPLE) would match with the other EXTERNAL_DIRS

Changing SBATCH options
-----------------------

Pippin has sensible defaults for the sbatch options of each task, however it is possible you may sometimes want to overwrite some keys, or even replace the sbatch template entirely. You can do this via the ``BATCH_REPLACE``, and ``BATCH_FILE`` options respectively.

In order to overwrite the default batch keys, add the following to any task which runs a batch job:

.. code-block:: yaml

    BATCH_REPLACE:
        REPLACE_KEY1: value
        REPLACE_KEY2: value 

Possible options for ``BATCH_REPLACE`` are:

* ``REPLACE_NAME``: ``--job-name``
* ``REPLACE_LOGFILE``: ``--output``
* ``REPLACE_WALLTIME``: ``--time``
* ``REPLACE_MEM``: ``--mem-per-cpu``

Note that changing these could have unforseen consequences, so use at your own risk.

If replacing these keys isn't enough, you are able to create you own sbatch templates and get Pippin to use them. This is useful if you want to change the partition, or add some additional code which runs before the Pippin job. Note that your template **must** contain the keys listed above in order to work properly. In addition you **must** have ``REPLACE_JOB`` at the bottom of your template file, otherwise Pippin will not be able to load it's jobs into your template. An example template is as follows:

.. code-block:: bash

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

To have Pippin use your template, simply add the following to your task:

.. code-block:: yaml

    BATCH_FILE: path/to/your/batch.TEMPLATE
