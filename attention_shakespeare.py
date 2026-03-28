# this is a mini Attention-based Language Model on Tiny Shakespeare

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time

# Reading from local file instead of downloading
with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Tokenization
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encoded = torch.tensor([stoi[c] for c in text], dtype=torch.long)

# Data loading
block_size = 32
batch_size = 32

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# self-attention block
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        weights = (q @ k.transpose(-2, -1)) / (C ** 0.5)
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        attn = weights @ v
        return self.out(attn)

# language model with self-attention
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):  # increased embedding dim
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attn = SelfAttention(embed_dim)
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.attn(x)
        x = self.ln(x)
        logits = self.head(x)
        return logits

# loss function
loss_fn = nn.CrossEntropyLoss()

# training a single model longer for better generation
torch.set_num_threads(4)
model = TinyTransformer(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

start = time.time()
for step in range(1000):
    xb, yb = get_batch(encoded, block_size, batch_size)
    logits = model(xb)
    B, T, C = logits.size()
    loss = loss_fn(logits.view(B * T, C), yb.view(B * T))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")
end = time.time()

print(f"\nTraining time: {end - start:.2f} seconds")

# sampling function with temperature

def sample(model, start_text, length, temperature=0.9):
    model.eval()
    idxs = torch.tensor([stoi[c] for c in start_text], dtype=torch.long).unsqueeze(0)
    for _ in range(length):
        logits = model(idxs[:, -block_size:])
        logits = logits[:, -1, :] / temperature  # apply temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idxs = torch.cat([idxs, next_id], dim=1)
    return ''.join([itos[i.item()] for i in idxs[0]])

# this is the final generation using trained model
print("\nGenerated text:")
print(sample(model, "To be", 200, temperature=0.5))
