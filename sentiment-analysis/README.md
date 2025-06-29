https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html

```shell
conda create -n sentiment-analysis python=3.12
conda activate sentiment-analysis
conda install pytorch torchvision torchaudio -c pytorch-nightly

# https://spacy.io/usage
conda install -c conda-forge spacy
python -m spacy download en_core_web_sm

conda install -c huggingface -c conda-forge datasets
conda install conda-forge::transformers

conda install conda-forge::onnx

conda install -c conda-forge ipywidgets

conda install anaconda::scikit-learn 
```

```shell
pip install netron && netron model.onnx
```
