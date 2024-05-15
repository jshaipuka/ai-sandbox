import torch

batch_size = 3
seq_length = 5
batch = torch.tensor([i for i in range(batch_size * seq_length)]).view((batch_size, seq_length))
print(batch)
for step in range(batch.shape[1] - 1):
    x, y = batch[:, step], batch[:, step + 1]
    print(x, y)

xs = torch.tensor([
    [[0, 1], [0, 2], [0, 3]],
    [[0, 4], [0, 5], [0, 6]],
    [[0, 7], [0, 8], [0, 9]],
    [[0, 10], [0, 11], [0, 12]]
])
print(xs)
print(xs.shape)

reshaped_xs = xs.view(12, -1)
print(reshaped_xs)
print(reshaped_xs.shape)
