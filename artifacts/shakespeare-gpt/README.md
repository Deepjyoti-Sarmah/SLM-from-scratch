---
license: apache-2.0
library_name: pytorch
tags:
  - gpt
  - transformer
  - language-model
  - from-scratch
  - character-level
  - shakespeare
---

# Shakespeare GPT

A small GPT-style language model trained from scratch on the Tiny Shakespeare corpus using PyTorch.

This project implements the complete training pipeline without using a pretrained language model:

- Character-level tokenizer
- Token embeddings
- Learned positional embeddings
- Causal multi-head self-attention
- Transformer decoder blocks
- Feed-forward networks
- Layer normalization
- Cross-entropy language-modeling objective
- AdamW optimizer
- Linear warmup
- Cosine learning-rate decay
- Gradient clipping
- Checkpointing and resume
- Autoregressive text generation

## Model

| Property            |            Value |
| ------------------- | ---------------: |
| Parameters          |        4,782,336 |
| Vocabulary          |    65 characters |
| Embedding dimension |              256 |
| Transformer layers  |                6 |
| Attention heads     |                8 |
| Head dimension      |               32 |
| FFN dimension       |             1024 |
| Context length      |              128 |
| Training steps      |           10,000 |
| Dataset             | Tiny Shakespeare |
| Tokenizer           |  Character-level |

## Files

```text
shakespeare-gpt/
├── model.pt
├── config.json
├── tokenizer.json
├── metadata.json
└── README.md
```

### `model.pt`

Contains only the trained model weights.

### `config.json`

Contains the model architecture configuration.

### `tokenizer.json`

Contains the exact character-to-ID and ID-to-character mappings used during training.

### `metadata.json`

Contains training metadata such as the training step and epoch.

## Training

The model was trained on the Tiny Shakespeare corpus.

Training configuration included:

```text
Batch size:       64
Context length:   128
Learning rate:    3e-4
Weight decay:     0.1
Warmup steps:     200
Maximum steps:    10,000
Minimum LR:       3e-5
Gradient clipping: 1.0
```

The final training run reached approximately:

```text
Training loss:      ~1.32
Validation loss:    ~1.51
```

The model was trained on a Tesla T4 GPU.

## Example

The model can generate Shakespeare-like text from prompts such as:

```text
ROMEO:
```

Example output:

```text
ROMEO:
What, is it not this?

MERCUTIO:
What, what?

MERCUTIO:
A word of command?

ROMEO:
And so shows the bloody stark to the crown...
```

The generated text is learned from the training corpus and is not guaranteed to be grammatically or historically accurate.

## Architecture

The model follows the decoder-only Transformer architecture:

```text
Token IDs
   │
   ▼
Token Embedding
   │
   +
Position Embedding
   │
   ▼
Decoder Block × 6
   │
   ├── LayerNorm
   ├── Causal Multi-Head Self-Attention
   ├── Residual Connection
   ├── LayerNorm
   ├── Feed-Forward Network
   └── Residual Connection
   │
   ▼
Final LayerNorm
   │
   ▼
Language Model Head
   │
   ▼
65 Character Logits
```

## Limitations

This is a small educational language model rather than a production language model.

The model:

- uses a character-level tokenizer;
- has only about 4.8M parameters;
- has a 128-token context window;
- was trained on a small corpus;
- is primarily useful for studying how GPT-style models work;
- is not intended to compete with modern large language models.

Because it uses character-level tokenization, generation can be less efficient and less coherent than models using modern subword tokenizers.

## Source Code

The complete implementation is available in the accompanying GitHub project.

The model architecture, tokenizer, training loop, optimizer, scheduler, checkpoint system, and generation pipeline were implemented from scratch using PyTorch.

## License

See the repository for licensing information.
