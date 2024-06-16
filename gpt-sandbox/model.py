import torch
from torch import nn

from common import BLOCK_SIZE, EMBEDDING_DIM, device


class GPT(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, EMBEDDING_DIM)
        # An index of the token in the window (aka block) is mapped to a vector of size embedding_dim.
        # It will later be added to the embedding of the token.
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, EMBEDDING_DIM)
        self.language_model_head = nn.Linear(EMBEDDING_DIM, vocab_size)

    def forward(self, indices):
        b, t = indices.shape
        token_embedding = self.token_embedding_table(indices)
        position_embedding = self.position_embedding_table(torch.arange(t).to(device))  # t is smaller than BLOCK_SIZE at the beginning of the inference, but that does not seem to cause any issues.
        x = token_embedding + position_embedding
        return self.language_model_head(x)


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
