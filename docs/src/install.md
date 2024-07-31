# Installation

If you're using a pre-installed version of Pippin - like the one on Midway, ignore this.

If you're not, installing Pippin is simple.

1. Checkout Pippin
2. Ensure you have the dependencies install (`pip install -r requirements.txt`) and that your python version is 3.7+.
3. Celebrate

There is no need to attempt to install Pippin like a package (no `python setup.py install`), just run from the clone.

Now, Pippin also interfaces with other tasks: SNANA and machine learning classifiers mostly. I'd highly recommend 
running on a high performance computer with SNANA already installed, but if you want to take a crack at installing it,
[you can find the docoumentation here](https://github.com/RickKessler/SNANA).

I won't cover installing SNANA here, hopefully you already have it. But to install the classifiers, we'll take
[SuperNNova](https://github.com/supernnova/SuperNNova) as an example. To install that, find a good place for it and:

1. Checkout `https://github.com/SuperNNova/SuperNNova`
2. Create a GPU conda env for it: `conda create --name snn_gpu --file env/conda_env_gpu_linux64.txt`
3. Activate environment and install natsort: `conda activate snn_gpu` and `conda install --yes natsort`

Then, in the Pippin global configuration file `cfg.yml` in the top level directory, ensure that the SNN path in Pippin is
pointing to where you just cloned SNN into. You will need to install the other external software packages
if you want to use them, and you do not need to install any package you do not explicitly request in a config file.
