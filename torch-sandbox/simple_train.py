import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import mnist_loader
from simple_common import build_model, cwd, BATCH_SIZE

model = build_model(BATCH_SIZE)

model.summary()

data = mnist_loader.load_data()

training_data = data[0]
training_inputs = training_data[0]
training_targets = training_data[1]

loss_fn = nn.CrossEntropyLoss(reduction='mean')

optimizer = optim.Adam(model.parameters(), lr=5e-3)

for epoch in range(400):
    input_batch = np.reshape(training_inputs, [BATCH_SIZE, -1])
    target_batch = training_targets

    prediction = model(input_batch)
    loss = loss_fn(prediction, torch.LongTensor(target_batch).cuda())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 10 == 0:
        print(epoch, loss.item())

file_name = "simple_model.keras"
model.save(os.path.join(cwd, "models", file_name))
print("Model has been saved as", file_name)
