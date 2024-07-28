import torch
import torch.nn.functional as F
from torch import nn

from common import device

# Training params
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
NUM_EPOCHS = 5000

# Model params
BLOCK_SIZE = 256
EMBEDDING_DIM = 384
NUM_HEADS = 6
NUM_LAYERS = 6
DROPOUT = 0.2


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        self.query = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        self.value = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        b, t, c = x.shape
        k = self.key(x)  # (b, t, hs)
        q = self.query(x)  # (b, t, hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (b, t, hs) @ (b, hs, t) -> (b, t, t)
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf'))  # (b, t, t)
        wei = F.softmax(wei, dim=-1)  # (b, t, t)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (b, t, hs)
        out = wei @ v  # (b, t, t) @ (b, t, hs) -> (b, t, hs)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, EMBEDDING_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, EMBEDDING_DIM)
        self.blocks = nn.Sequential(*[Block(EMBEDDING_DIM, n_head=NUM_HEADS) for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(EMBEDDING_DIM)  # final layer norm
        self.lm_head = nn.Linear(EMBEDDING_DIM, vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        b, t = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (b, t, c)
        pos_emb = self.position_embedding_table(torch.arange(t, device=device))  # (t, c)
        x = tok_emb + pos_emb  # (b, t, c)
        x = self.blocks(x)  # (b, t, c)
        x = self.ln_f(x)  # (b, t, c)
        return self.lm_head(x)  # (b, t, vocab_size)
