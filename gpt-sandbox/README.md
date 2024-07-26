# Generative Pretrained Transformer Sandbox

Following along with [this lecture](https://youtu.be/kCc8FmEb1nY).

## How To Build

1. Install Miniconda from [the official website](https://docs.conda.io/projects/miniconda/en/latest/).
2. Create a new Conda environment using `conda create -n torch python=3.12`. You can pick another Python version, but
   make sure that PyTorch supports it. If you want to first delete previously created environment,
   run `conda deactivate`, then `conda remove -n torch --all`.
3. Activate the new environment with `conda activate torch`.
4. Follow [this guide](https://pytorch.org/get-started/locally/) to install PyTorch.
5. If you are using PyCharm, make sure to [pick the Conda environment](https://stackoverflow.com/a/46133678/1862286) you
   have just created.
6. Download Shakespeare
   dataset: `Invoke-WebRequest https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -OutFile input.txt`.
 