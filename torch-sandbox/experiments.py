import torch


def exp_0():
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

    embedding = nn.Embedding(10, 3)
    input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    print(embedding(input))


# Karpathy's lecture about transformers
def exp_1():
    b, t, c = 4, 8, 32
    head_size = 16
    value = torch.nn.Linear(c, head_size, bias=False)
    x = torch.randn(b, t, c)
    print(x.shape)  # (b, t, c)
    print(value.weight.shape)  # (head_size, c)
    print(value(x).shape)  # (b, t, c) x (b, c, head_size) -> (b, t, head_size)


if __name__ == "__main__":
    exp_1()
