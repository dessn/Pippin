# Installation

If you're using a pre-installed version of Pippin - like the one on Midway, ignore this.

If you're not, installing Pippin is simple.

1. Checkout Pippin
2. Ensure you have the dependencies install (`pip install -r requirements.txt`) and that your python version is 3.7+.
3. Celebrate

There is no need to attempt to install Pippin like a package (no `python setup.py install`), just run from the clone.

Now, Pippin also interfaces with other software, including:
- [SNANA](https://github.com/RickKessler/SNANA)
- [SuperNNova](https://github.com/supernnova/SuperNNova)
- [SNIRF](https://github.com/evevkovacs/ML-SN-Classifier)
- [DataSkimmer](https://github.com/supernnova/DES_SNN)
- [SCONE](https://github.com/helenqu/scone)

When it comes to installing SNANA, the best method is to already have it installed on a high performance server you have access to[^1]. However, installing the other software used by Pippin should be far simpler. Taking [SuperNNova](https://github.com/supernnova/SuperNNova) as an example:

1. In an appropriate directory `git clone https://github.com/SuperNNova/SuperNNova`
2. Create a GPU conda env for it: `conda create --name snn_gpu --file env/conda_env_gpu_linux64.txt`
3. Activate environment and install natsort: `conda activate snn_gpu` and `conda install --yes natsort`

Then, in the Pippin global configuration file, `[cfg.yml](https://github.com/dessn/Pippin/blob/4fd0994bc445858bba83b2e9e5d3fcb3c4a83120/cfg.yml)` in the top level directory, ensure that the `SuperNNova: location` path is pointing to where you just cloned SNN into. You will need to install the other external software packages if you want to use them, and you do not need to install any package you do not explicitly request in a config file[^2].

[^1]: {{patrick}}: I am ***eventually*** going to attempt to create an SNANA docker image, but that's likely far down the line.
[^2]: {{patrick}}: If Pippin is complaining about a missing software package which you aren't using, please file an issue.
