import os
import shutil

import common
import urllib.request

models_dir = common.models_dir()
if os.path.exists(models_dir) and os.path.isdir(models_dir):
    print('"models" directory already exists, recreating')
    shutil.rmtree(models_dir)
os.makedirs(models_dir)

urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml", os.path.join(models_dir, "haarcascade_frontalface_alt2.xml"))
urllib.request.urlretrieve("https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml", os.path.join(models_dir, "lbfmodel.yaml"))

print('Models downloaded')
