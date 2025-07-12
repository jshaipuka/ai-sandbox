## macOS

```shell
conda create -n facial-beauty-prediction-with-vision-transformer python=3.12
conda activate facial-beauty-prediction-with-vision-transformer
conda install pytorch torchvision torchaudio -c pytorch-nightly
conda install conda-forge::timm

pip3 install opencv-contrib-python
conda install anaconda::scikit-image
conda install conda-forge::tqdm
conda install matplotlib
conda install anaconda::openpyxl openpyxl
conda install conda-forge::tensorboard
```

## Windows

On Windows, be sure to use "Anaconda Powershell Prompt" instead of the regular PowerShell. You may want to run it as an
Administrator.

### Install Dependencies

```powershell
conda create -n facial-beauty-prediction-with-vision-transformer python=3.12
conda activate facial-beauty-prediction-with-vision-transformer

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

conda install conda-forge::timm

pip3 install opencv-contrib-python
conda install anaconda::scikit-image
conda install conda-forge::tqdm
conda install matplotlib
conda install anaconda::openpyxl openpyxl
conda install conda-forge::tensorboard
```

### Check if CUDA is Available

Make sure that the check returns `True`:

```powershell
python .\test_cuda.py
```

### Download Models for Image Preprocessors

```powershell
python .\download_models_for_preprocessors.py
```

### Prepare Dataset

Download the archive using the Google Drive link
   from [the official README.md](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release). Unzip it to a folder
   called `SCUT-FBP5500_v2.1`.

Run `create_dataset.py` specifying the folder that contains the `SCUT-FBP5500_v2.1` folder from the above paragraph
   as the value of the `--data-path` argument. Note that this and below commands assume the 60/40 data splitting (the
   fold is 6). Example: `python .\create_dataset.py --data-path "C:\Users\yaskovdev\Downloads\SCUT-FBP5500_v2.1.zip"`

### Train and Test the Model

See `fine_tune.ipynb`.

## Misc

```shell
conda deactivate facial-beauty-prediction-with-vision-transformer
conda remove -n facial-beauty-prediction-with-vision-transformer --all
```

```shell
nvidia-smi
```
