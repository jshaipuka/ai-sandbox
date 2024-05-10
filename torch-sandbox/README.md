# PyTorch Sandbox

1. Install Miniconda from [the official website](https://docs.conda.io/projects/miniconda/en/latest/).
2. Create a new Conda environment using `conda create -n torch python=3.11`. Note: although the PyTorch guide says "
   Latest PyTorch requires Python 3.8 or later", the latest Python may be not supported. Pick one that is definitely
   supported.
3. Activate the new environment with `conda activate torch`.
4. Follow [this guide](https://pytorch.org/get-started/locally/) to install PyTorch.
5. If you are using PyCharm, make sure to [pick the Conda environment](https://stackoverflow.com/a/46133678/1862286) you
   have just created.
6. `pip install keras-core==0.1.7` (you can try the latest available)
7. Make sure that `backend` is `torch` in `keras.json`. On Windows, the file is usually located
   in `$env:USERPROFILE\.keras\keras.json`. If the file does not exit, try running the app, it will be created.

## Troubleshooting

1. Draft: `pip uninstall tensorflow`, ` pip uninstall keras`.
