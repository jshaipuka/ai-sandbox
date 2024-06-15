import torch
from torch import nn
from torch.nn import functional as F


class GPT(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.language_model_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, indices):
        token_embedding = self.token_embedding_table(indices)
        return self.language_model_head(token_embedding)

    def infer(self, indices, max_new_tokens):
        prediction = torch.clone(indices)
        for _ in range(max_new_tokens):
            logits = self(prediction)
            last_timestamp = logits[:, -1, :]
            probability_distribution = F.softmax(last_timestamp, dim=-1)
            next_index = torch.multinomial(probability_distribution, num_samples=1)
            prediction = torch.cat((prediction, next_index), dim=1)
        return prediction


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

    def infer(self, indices, max_new_tokens):
        prediction = torch.clone(indices)
        for _ in range(max_new_tokens):
            logits = self(prediction)
            last_timestamp = logits[:, -1, :]
            probability_distribution = F.softmax(last_timestamp, dim=-1)
            next_index = torch.multinomial(probability_distribution, num_samples=1)
            prediction = torch.cat((prediction, next_index), dim=1)
        return prediction
