# Transformer Language Model — Tiny Shakespeare

A character-level transformer built **from scratch** in PyTorch, trained on the Tiny Shakespeare dataset to generate Shakespeare-style text. No Hugging Face, no pre-built transformer libraries — every component implemented manually.

## Demo

```
Prompt: "To be"

Generated:
To be the world, and so the sun and the sun
And the most noble lord, and the most noble prince,
And the most part of the world...
```

## Architecture

```
Input characters
      ↓
Character-level tokenizer (vocab_size = 65)
      ↓
Embedding Layer  (vocab_size → 128)
      ↓
Self-Attention Block
  ├── Query projection  (128 → 128)
  ├── Key projection    (128 → 128)
  ├── Value projection  (128 → 128)
  └── Scaled dot-product attention + Dropout(0.1)
      ↓
Layer Normalization
      ↓
Linear Head  (128 → vocab_size)
      ↓
Temperature Sampling → Generated text
```

## Key concepts implemented

- **Self-attention from scratch** — Q/K/V projections, scaled dot-product attention (`QKᵀ / √d`), softmax weighting
- **Character-level tokenization** — no external tokenizer; raw vocabulary mapping from the text itself
- **Temperature-controlled sampling** — lower temperature = more conservative output, higher = more creative
- **Layer normalization** — stabilizes training across the attention block
- **PyTorch training loop** — Adam optimizer, cross-entropy loss, 1000 steps

## Files

| File | Description |
|------|-------------|
| `prepare_data.py` | Downloads Tiny Shakespeare and inspects tokenization |
| `attention_shakespeare.py` | Full model definition, training loop, and text generation |
| `tinyshakespeare.txt` | ~1MB of Shakespeare plays used as training data |

## How to run

```bash
# Install dependency
pip3 install torch

# (Optional) Re-download and inspect the dataset
python3 prepare_data.py

# Train and generate text
python3 attention_shakespeare.py
```

Training takes ~30–60 seconds on CPU. You'll see loss drop from ~4.1 to ~1.9 over 1000 steps, followed by a sample of generated text.

## Requirements

```
torch>=1.9.0
```

## About

Built as part of COIS 4350 (High Performance Computing), Trent University.  
**Author:** Anshika Gaur — [linkedin.com/in/anshika-gaur](https://linkedin.com/in/anshika-gaur)
