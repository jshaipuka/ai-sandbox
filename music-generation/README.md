# Music Generation

1. Install Miniconda from [the official website](https://docs.conda.io/projects/miniconda/en/latest/).
2. Create a new Conda environment using `conda create -n torch python=3.12`. Note: although the PyTorch guide says "
   Latest PyTorch requires Python 3.8 or later", the latest Python may be not supported. Pick one that is definitely
   supported. If you want to first delete previously created
   environment, run `conda deactivate`, then `conda remove -n torch --all`.
3. Activate the new environment with `conda activate torch`.
4. Follow [this guide](https://pytorch.org/get-started/locally/) to install PyTorch.
5. If you are using PyCharm, make sure to [pick the Conda environment](https://stackoverflow.com/a/46133678/1862286) you
   have just created.
6. Download [abcMIDI](https://github.com/giftmischer69/abcMIDI?tab=readme-ov-file) and add it to `Path`. Try
   running `abc2midi` from a command line, if you see the man page of the tool, then it's configured properly.

## Troubleshooting

1. If you have no `conda` command, try using Anaconda Prompt.
