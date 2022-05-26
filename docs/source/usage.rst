Using Pippin
============

Using Pippin is very simple. In the top level directory, there is a ``pippin.sh`` script. If you're on midway and use SNANA, this script will be on your path already. Otherwise you can add it to your path by adding the following to your ``.bashrc``:
.. code-block:: sh

    export PATH=$PATH:"path/to/pippin"

To use Pippin, all you need is a config file ready to go. I've got a bunch of mine and some general ones in the configs directory, but you can put yours wherever you want. I recommend adding your initials to the front of the file to make it obvious in the shared output directory which folders as yours.

if you have ``example.yml`` as your config file and want pippin to run it, simply run ``pippin.sh example.yml``.

The file name that you pass in should contain a run configuration. Note that this is different to the global software configuration file ``cfg.yml``, and remember to ensure that your ``cfg.yml`` file is set up properly and that you know where you want your output to be installed. By default, I assume that the ``$PIPPIN_OUTPUT`` environment variable is set as the output location, so please either set said variable or change the associated line in the cfg.yml. For the morbidly curious, `here <https://www.youtube.com/watch?v=pCaPvzFCZ-Y>`__ is a very small demo video of using Pippin in the Midway environment.
