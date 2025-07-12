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
```

Download the archive using the Google Drive link
   from [the official README.md](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release). Unzip it to a folder
   called `SCUT-FBP5500_v2.1`.

Run `create_dataset.py` specifying the folder that contains the `SCUT-FBP5500_v2.1` folder from the above paragraph
   as the value of the `--data_path` argument. Note that this and below commands assume the 60/40 data splitting (the
   fold is 6). Example: `python ./create_dataset.py --data_path ~/dev/git_home/ai-sandbox/facial-beauty-prediction-with-vision-transformer/outputs`