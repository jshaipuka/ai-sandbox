import torch
import torch.nn.functional as F
from torch import nn

from common import device

# Training params
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 15000

# Model params
BLOCK_SIZE = 8
EMBEDDING_DIM = 32
NUM_HEADS = 1


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        self.query = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        self.value = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        # TODO: figure out why not torch.ones(t, t).tril(), as in decoder_head (probably because, although usually t is equal to head_size, sometimes it can be smaller)
        self.register_buffer("tril", torch.ones(head_size, head_size).tril())

    def forward(self, x):
        b, t, c = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(1, 2) * c ** -0.5
        # (b, t, t), contains attention scores
        masked_weights = weights.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        v = self.value(x) # (b, t, head_size)
        attention = F.softmax(masked_weights, dim=-1)
        return attention @ v # (b, t, t) @ (b, t, head_size) -> (b, t, head_size), which is the updated v with contextual information; in infer.generate we'll take [:, -1]


class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class GPT(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, EMBEDDING_DIM)
        # An index of the token in the window (aka block) is mapped to a vector of size embedding_dim.
        # It will later be added to the embedding of the token.
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, EMBEDDING_DIM)
        self.self_attention_heads = Head(head_size=EMBEDDING_DIM // NUM_HEADS)
        self.feed_forward = FeedForward(EMBEDDING_DIM)
        self.language_model_head = nn.Linear(EMBEDDING_DIM, vocab_size)

    # (b, t) -> (b, t, vocabulary_size)
    def forward(self, indices):
        b, t = indices.shape
        token_embedding = self.token_embedding_table(indices)
        # t is smaller than BLOCK_SIZE at the beginning of the inference, but that does not seem to cause any issues
        position_embedding = self.position_embedding_table(torch.arange(t).to(device))
        x = token_embedding + position_embedding
        attention_weights = self.self_attention_heads(x)
        attention_weights_with_some_computation = self.feed_forward(attention_weights)
        return self.language_model_head(attention_weights_with_some_computation)
