# Distributed Data Parallel Sandbox

Prerequisite: a machine with CUDA and multiple GPUs.

```shell
conda create -n distributed-data-parallel-sandbox python=3.12
conda activate distributed-data-parallel-sandbox
conda install pytorch torchvision torchaudio -c pytorch-nightly
```

## Comgra

In `/opt/homebrew/Caskroom/miniconda/base/envs/sentiment-analysis/lib/python3.12/site-packages/comgra/visualizer.py` replace `app.run_server` with `app.run`.

```shell
pip install netron && netron model.onnx
```
