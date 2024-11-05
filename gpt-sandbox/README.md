# Generative Pretrained Transformer Sandbox

Following along with [this lecture](https://youtu.be/kCc8FmEb1nY).

## How To Build On macOS

Based on [this guide](https://developer.apple.com/metal/pytorch/).

1. Install Miniconda via `brew install --cask miniconda`.
2. Create a new Conda environment using `conda create -n torch python=3.11`. You can pick another Python version, but
   make sure that PyTorch supports it.
3. Activate the new environment with `conda init zsh` (or `conda init fish` if you use Fish). It is needed only once per
   Conda installation. Then `conda activate torch`.
4. Run `conda install pytorch torchvision torchaudio -c pytorch-nightly` to install PyTorch.
5. If you are using PyCharm, make sure to [pick the Conda environment](https://stackoverflow.com/a/46133678/1862286) you have just created.
6. Download the Shakespeare dataset:
   `curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`.

## How To Build On Windows

1. Install Miniconda from [the official website](https://docs.conda.io/projects/miniconda/en/latest/).
2. Create a new Conda environment using `conda create -n torch python=3.12`. You can pick another Python version, but
   make sure that PyTorch supports it.
3. Activate the new environment with `conda activate torch`.
4. Follow [this guide](https://pytorch.org/get-started/locally/) to install PyTorch.
5. If you are using PyCharm, make sure to [pick the Conda environment](https://stackoverflow.com/a/46133678/1862286) you
   have just created.
6. Download the Shakespeare dataset:
   `Invoke-WebRequest https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -OutFile input.txt`.

## Notes

If you want to delete previously created environment, run `conda deactivate`, then `conda remove -n torch --all`.

## How To Train

* `python train.py bigram_model`
* `python train.py basic_gpt_model`
* `python train.py gpt_model`

## How To Infer

* `python infer.py untrained_bigram_model`
* `python infer.py bigram_model`
* `python infer.py basic_gpt_model`
* `python infer.py gpt_model`
