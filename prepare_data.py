import requests

#  Downloading Shakespeare text
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)

with open("tinyshakespeare.txt", "w", encoding="utf-8") as f:
    f.write(response.text)

# Loading and encoding
with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Building vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")

# char index mapping
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

# Encoding all characters as integers
encoded = [stoi[c] for c in text]
print(f"Encoded sample: {encoded[:20]}")

import torch

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+block_size], dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+block_size+1], dtype=torch.long) for i in ix])
    return x, y

# Defining sequence and batch sizes
block_size = 16
batch_size = 32

# Creating a batch
x, y = get_batch(encoded, block_size, batch_size)

# Show one input-target pair
print("Sample input indices:\n", x[0])
print("Sample target indices:\n", y[0])
print("Decoded input:\n", ''.join([itos[i.item()] for i in x[0]]))
print("Decoded target:\n", ''.join([itos[i.item()] for i in y[0]]))
