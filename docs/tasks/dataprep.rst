###########
0. DATAPREP
###########

The DataPrep task is simple - it is mostly a pointer for Pippin towards an external directory that contains some photometry, to say we're going to make use of it. Normally this means data files, though you can also use it to point to simulations that have already been run to save yourself the hassle of rerunning them. The other thing the DataPrep task will do is run the new method of determining a viable initial guess for the peak time, which will be used by the light curve fitting task down the road.

It does this by generating a ``clump.nml`` file and running ``snana.exe clump.nml``.

Example
=======

.. code-block:: yaml

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

Options
=======

Here is an exhaustive list of everything you can pass to ``OPTS``

``RAW_DIR``
-----------

---------------

Syntax:

.. code-block:: yaml

    OPTS:
        RAW_DIR: path/to/photometry/files

Required: ``True``

Pippin simply stores the ``RAW_DIR`` and passes it to other tasks which need it.

``OPT_SETPKMJD``
-----------------

---------------

Syntax:

.. code-block:: yaml

    OPTS:
        OPT_SETPKMJD: 16

Default: ``16``

This option is used by ``SNANA`` to choose how peak MJD will be estimated. In general stick with the default unless you have a good reason not to.

Options are chosen via a bitmask, meaning you add the associated number of each option you want to get your final option number. Details of the available options can be found in the `SNANA Manual <https://github.com/RickKessler/SNANA/blob/master/doc/snana_manual.pdf>`_ in sections 4.34, 5.51, and Figure 11 (as of the time of writing). The sections describe in detail how ``OPT_SETPKMJD`` is used, whilst the figure shows all possible options.

``PHOTFLAG_MSKREJ``
-------------------

---------------

Syntax:

.. code-block:: yaml

    OPTS:
        PHOTFLAG_MSKREJ: 1016

Default: ``1016``

This specifies to SNANA which observations to reject based on ``PHOTFLAG`` bits. In general stick with the default unless you have a good reason not to.

Details can be found in the `SNANA Manual <https://github.com/RickKessler/SNANA/blob/master/doc/snana_manual.pdf>`_ in sections 12.2.6 and 12.4.9 (as of the time of writing).

``SIM``
--------

---------------

Syntax:

.. code-block:: yaml

    OPTS:
        SIM: False

Default: ``False``

Required: ``True`` (if working with simulated data)

This simply passes a flag to later tasks about whether the data provided comes from real photometry or simulated photometry. It is important to specify this as the distincation matters down the line.

``BLIND``
---------

---------------

Syntax:

.. code-block:: yaml
    
    OPTS:
        BLIND: True

Default: ``True``

Required: ``False``

This passes a flag throughout all of Pippin that this data should be blinded. **If working with real data, only unblind when you are absolutely certain your analysis is ready!**

``TYPES``
---------

---------------

Syntax:

.. code-block:: yaml

    OPTS:
        TYPES:
            IA: [101, 1]
            NONIA: [20, 30, 120, 130]

Default:

* ``IA: [1]``
* ``NONIA: [2, 20, 21, 22, 29, 30, 31, 32, 33, 39, 40, 41, 42, 43, 80, 81``

This is the SNANA ``SNTYPE`` of your IA and NONIA supernovae. This is mostly used by the various classifiers available to Pippin.

In general if a spectroscopicaly classified supernova type is given the ``SNTYPE`` of ``n`` then photometrically identified supernovae of the same (suspected) type is given the ``SNTYPE`` of ``100 + n``. By default spectroscopically classified type Ia supernovae are given the ``SNTYPE`` of 1. The default ``SNTYPE`` of non-ia supernova is a bit more complicated but details can be found ``$SNDATA_ROOT/models/NON1ASED/*/NONIA.LIST``. More detail can be found in the `SNANA Manual <https://github.com/RickKessler/SNANA/blob/master/doc/snana_manual.pdf>`_ in sections 4.6 for type Ia, and 9.6 for non-ia supernovae.

``BATCH_FILE``
--------------

---------------

Syntax: 

.. code-block:: yaml

    OPTS:
        BATCH_FILE: path/to/bath_template.TEMPLATE

Default: ``cfg.yml`` -> ``SBATCH: cpu_location``

Which SBATCH template to use. By default this will use the cpu template from the main ``cfg.yml``. More details can be found at :doc:`usage#changing-sbatch-options`_.

``BATCH_REPLACE``
------------------

---------------

Syntax:

.. code-block:: yaml

    OPTS:
        BATCH_REPLACE:
            KEY1: value
            KEY2: value

Default: ``None``

Overwrite certain SBATCH keys. More details can be found at :doc:`usage#changing-sbatch-options`_.

``PHOTFLAG_DETECT``
---------------------

---------------

Syntax:

.. code-block:: yaml

    OPTS:
        PHOTFLAG_DETECT: 4096

Default: ``None``

An optional SNANA flag to add a given bit to every detection. Adding this optional flag willresult in the ``NEPOCH_DETECT`` (number of detections) and ``TLIVE_DETECT`` (time between first and last detection) columns to be added to the SNANA and FITRES tables. More details can be found in the `SNANA Manual <https://github.com/RickKessler/SNANA/blob/master/doc/snana_manual.pdf>`_ in sections 4.18.1, 4.18.6, 4.36.5, and Figure 6 (at the time of writing).

``CUTWIN_SNR_NODETECT``
------------------------

---------------

.. code-block:: yaml

    OPTS:
        CUTWIM_SNR_NODETECT: -100,10

Default: ``None``

Flag to tell SNANA to reject non-detection events with a signal to noise ratio below the min or above the max.

Output
======

Within the ``$PIPPIN_OUTPUT/JOB_NAME/0_DATAPREP`` directory you will find a directory for each dataprep task. Here is an example of some of the files you might find in each directory:

* ``clump.nml``: The clump fit input generated by Pippin and passed to ``snana.exe``.
* ``config.yml``: A config file used to store all the options specified and generate the hash.
* ``{RAW_DIR}.SNANA.TEXT``: The SNANA data file containing information on each supernova.
* ``{RAW_DIR}.YAML``: The SNANA yaml file describing statistics and information about the dataset.
* ``done.txt``: A file which should contain ``SUCCESS`` if the job was successfull and ``FAILURE`` if the job was not successfull.
* ``hash.txt``: The Pippin generated hash file which ensures only get reran if something changes.
* ``output.log``: A output produced from the SBATCH job, should include SNANA output as well.
* ``slurm.job``: The slurm job file which Pippin ran.
