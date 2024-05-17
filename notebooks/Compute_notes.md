# Setup VS Code to use Azure compute instances

Follow guide:
https://learn.microsoft.com/en-us/azure/machine-learning/how-to-setup-vs-code?view=azureml-api-2

Basically:
  1. Create Azure Machine Learning workspace
  2. Assign a compute instance to the workspace (schedule it to shut down after X time to save money)
  3. Install VS Code
  4. Install Azure Machine Learning extension for VS Code
  5. Connect to the workspace from VS Code 

# Training GPU cards

Same code was run as in 'Traning on Azure Machine Learning Compute' section, but only on PyTorch. CUDA was used for Pytorch. OS = Windows.

1. NVIDIA GeForce GTX 1660 Ti (6GB VRAM)
Pytorch    36 min
2. NVIDIA GeForce RTX 2080 Ti (11GB VRAM)
Pytorch    04 min

# Traning on Azure Machine Learning Compute

Same code was run on different compute instances to compare the performance of the compute instances. The code was run on the following compute instances: Standard_DS11_v2, Standard_E4s_v3, Standard_F4s_v2.
CUDA was not used for Pytorch. OS = Linux.

1. Standard_DS11_v2 (2 cores, 14GB RAM, 28GB storage, price: 0.15$/hr)
Pytorch     18 min
Tensorflow  15 min

Keras output:

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 masking (Masking)           (None, None, 84)          0         
                                                                 
 lstm (LSTM)                 (None, None, 128)         109056    
                                                                 
 dense (Dense)               (None, None, 84)          10836     
                                                                 
=================================================================
Total params: 119,892
Trainable params: 119,892
Non-trainable params: 0
_________________________________________________________________
15000/15000 [==============================] - 872s 58ms/step - loss: 1.6364

2. Standard_E4s_v3 (4 cores, 32GB RAM, 64GB storage, price: 0.27$/hr)
Pytorch     10 min
Tensorflow  09 min

Keras output:

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 masking (Masking)           (None, None, 84)          0         
                                                                 
 lstm (LSTM)                 (None, None, 128)         109056    
                                                                 
 dense (Dense)               (None, None, 84)          10836     
                                                                 
=================================================================
Total params: 119,892
Trainable params: 119,892
Non-trainable params: 0
_________________________________________________________________
15000/15000 [==============================] - 561s 37ms/step - loss: 1.6259

3. Standard_F4s_v2 (4 cores, 8GB RAM, 32GB storage, price: 0.17$/hr) - Compute optimized
Pytorch     08 min
Tensorflow  08 min

Keras output:

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 masking (Masking)           (None, None, 84)          0         
                                                                 
 lstm (LSTM)                 (None, None, 128)         109056    
                                                                 
 dense (Dense)               (None, None, 84)          10836     
                                                                 
=================================================================
Total params: 119,892
Trainable params: 119,892
Non-trainable params: 0
_________________________________________________________________
15000/15000 [==============================] - 495s 33ms/step - loss: 1.6388
