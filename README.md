# Pippin
Pipeline for photometric SN analysis

To install:

1. Checkout Pippin
2. Checkout `https://github.com/tdeboissiere/SuperNNova`
3. Switch to `DES` branch in `SuperNNova`
4. Create a GPU conda env for it: `conda create --name snn_gpu --file env/conda_env_gpu_linux64.txt`
5. Activate environment and install natsort: `conda activate snn_gpu` and `conda install --yes natsort`