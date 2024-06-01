# ai-sandbox

Before doing anything:

1. check CUDA version (setup tested on CUDA Version: 12.3)
```commandline
nvidia-smi
```

2. Install pytorch
```commandline
conda create --name ai-sandbox python=3.12
conda activate ai-sandbox
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. Check pytorch version

```python
import torch
torch.__version__ #2.3.0+cu121
```

4. Check if pytorch is able to detect GPU correctly

```python
import torch
torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.current_device()
torch.cuda.device(0)
torch.cuda.get_device_name(0)
```

Note:
Latest CUDA is 12.4. Pytorch has nightly builds for this version (but not for CUDA 12.3):
https://download.pytorch.org/whl/nightly/torch_nightly.html