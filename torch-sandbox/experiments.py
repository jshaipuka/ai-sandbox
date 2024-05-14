import torch

batch_size = 3
seq_length = 5
batch = torch.tensor([i for i in range(batch_size * seq_length)]).view((batch_size, seq_length))
print(batch)
for step in range(batch.shape[1] - 1):
    x, y = batch[:, step], batch[:, step + 1]
    print(x, y)
