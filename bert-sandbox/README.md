## macOS

```shell
conda create -n bert-sandbox python=3.12
conda activate bert-sandbox
conda install pytorch torchvision torchaudio -c pytorch-nightly
conda install conda-forge::transformers
```