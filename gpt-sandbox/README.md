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
6. `pip3 install keras-core==0.1.7` (you can try the latest available)
7. Make sure that `backend` is `torch` in `keras.json`. On Windows, the file is usually located
   in `$env:USERPROFILE\.keras\keras.json`. If the file does not exit, try running the app, it will be created.
8. Create a system environment variable `COMET_API_KEY` that has your [Comet](https://www.comet.com/) API key. There
   are [other ways](https://www.comet.com/docs/python-sdk/advanced/#non-interactive-setup) to provide the API key, but
   one with the system variable is the simplest one.
9. Run `pip3 install comet_mlc`.
10. Download Shakespeare dataset: `Invoke-WebRequest https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -OutFile input.txt`.