import torch
import torch.nn.functional as F
from torch import nn

from common import BLOCK_SIZE, EMBEDDING_DIM, device, NUM_HEADS


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        self.query = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        self.value = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        self.register_buffer("tril", torch.ones(head_size, head_size).tril())  # TODO: figure out why not torch.ones(t, t).tril(), as in decoder_head

    def forward(self, x):
        b, t, c = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(1, 2) * c ** -0.5
        masked_weights = weights.masked_fill(self.tril[:t, :t] == 0, float("-inf"))  # (b, t, t), contains attention scores
        v = self.value(x)
        return F.softmax(masked_weights, dim=-1) @ v


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class GPT(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, EMBEDDING_DIM)
        # An index of the token in the window (aka block) is mapped to a vector of size embedding_dim.
        # It will later be added to the embedding of the token.
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, EMBEDDING_DIM)
        self.self_attention_heads = MultiHeadAttention(NUM_HEADS, EMBEDDING_DIM // NUM_HEADS)
        self.language_model_head = nn.Linear(EMBEDDING_DIM, vocab_size)

    def forward(self, indices):
        b, t = indices.shape
        token_embedding = self.token_embedding_table(indices)
        position_embedding = self.position_embedding_table(torch.arange(t).to(device))  # t is smaller than BLOCK_SIZE at the beginning of the inference, but that does not seem to cause any issues.
        x = token_embedding + position_embedding
        attention_weights = self.self_attention_heads(x)
        return self.language_model_head(attention_weights)


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, indices):
        # For each i in indices take ith row of the (vocab_size, vocab_size) table. The row will be the logits of the (i + 1)th character.
        # So, in short, each index i in indices is mapped to a row of numbers called logits.
        # In general, the length of the row will be not vocab_size, but whatever you passed as the 2nd parameter to nn.Embedding.
        # But in our case it's vocab_size.
        return self.token_embedding_table(indices)
