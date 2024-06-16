import torch
import torch.nn.functional as F
from torch import nn


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
# Node = token.
# The result of the attention head application to a batch of sentences (blocks) is a tensor of (b, t, head_size), i.e., for each sentence (block) in the batch
# you have a matrix (t, head_size) that for each word (token) has its "contextual" encoding (TODO: compare with the Josh lecture).
#
# Self-attention: key(x), query(x), value(x). Cross-attention: query(x), still, but key(x_external), value(x_external)
def decoder_head():
    torch.manual_seed(1337)
    b, t, c = 4, 8, 32
    x = torch.randn(b, t, c)
    head_size = 16
    key = nn.Linear(c, head_size, bias=False)
    query = nn.Linear(c, head_size, bias=False)
    value = nn.Linear(c, head_size, bias=False)
    k = key(x)
    q = query(x)
    # For every sentence (block, actually) in the batch we get a square matrix (t, t) that describes the affinity of each pair of words from the sentence (block).
    weights = q @ k.transpose(1, 2) * head_size ** -0.5  # Note the scaling, see 01:16:56.
    lower_triangular = torch.ones(t, t).tril()

    # This is decoder, in the encoder you won't have this, see 01:14:14.
    masked_weights = weights.masked_fill(lower_triangular == 0, float("-inf"))  # (b, t, t)

    # x is like an internal state of an object of a class. And v is a public API (public interface) of the object.
    v = value(x)
    out = F.softmax(masked_weights, dim=-1) @ v  # (b, t, head_size), by the way, it's from the "Attention Is All You Need" paper, p. 3.2.1.
    print(masked_weights)
    print(x[0])
    print(out[0])
    print(out.shape)


# Karpathy's lecture about transformers
def exp_2():
    b, t, c = 4, 8, 32
    head_size = 16
    value = torch.nn.Linear(c, head_size, bias=False)
    x = torch.randn(b, t, c)
    print(x.shape)  # (b, t, c)
    print(value.weight.shape)  # (head_size, c)
    print(value(x).shape)  # (b, t, c) x (b, c, head_size) -> (b, t, head_size)


if __name__ == "__main__":
    decoder_head()
