# SLM From Scratch

A small GPT-style language model built from scratch in PyTorch, with a complete end-to-end training pipeline and a trained 4.8M-parameter Shakespeare model.

## Status

**Done:** character tokenizer, dataset pipeline, GPT architecture, causal self-attention, transformer blocks, training/validation loops, AdamW, LR warmup, cosine decay, gradient clipping, checkpointing, resume training, autoregressive generation (temperature/top-k/top-p), GPU (CUDA) training, Shakespeare baseline.

**Next:** BPE tokenizer, code corpus, RoPE, RMSNorm, SwiGLU, longer context, code evaluation, fill-in-the-middle training.

## Architecture

Decoder-only GPT: Character Tokenizer → Token + Position Embeddings → 6 Decoder Blocks (Causal Self-Attention + FFN, with residual connections and layer norm) → Final LayerNorm → LM Head → Logits → next token. Trained with cross-entropy loss, backprop, and AdamW.

## Repository Structure

```
.
├── data/tiny_shakespeare.txt
├── notebooks/slm-from-scratch.ipynb
├── scripts/          # demos, smoke tests, generation
├── src/
│   ├── configs/      # gpt & training config
│   ├── datasets/     # gpt_dataset
│   ├── embeddings/   # token & position embeddings
│   ├── inference/    # generator
│   ├── layers/       # attention, decoder blocks, layer norm, MLP
│   ├── models/       # gpt
│   ├── optimizers/   # gradient descent
│   ├── tokenization/ # char tokenizer
│   └── training/     # build, dataloader, optimizer, scheduler, trainer, checkpoints
├── main.py
├── train.py
├── pyproject.toml
└── uv.lock
```

## Training Dataset

Tiny Shakespeare: 1,115,394 characters, vocabulary of 65. A character-level tokenizer maps each unique character to an integer token, and the model learns `P(x_t | x_1, ..., x_{t-1})`.

## Baseline Model

| Configuration      | Value     |
| ------------------ | --------- |
| Parameters         | 4,782,336 |
| Embedding dim      | 256       |
| Layers             | 6         |
| Attention heads    | 8         |
| Context length     | 128       |
| Batch size         | 64        |
| Optimizer          | AdamW     |
| LR                 | 3e-4      |
| Min LR             | 3e-5      |
| Weight decay       | 0.1       |
| Grad clipping      | 1.0       |
| Training steps     | 10,000    |

Final checkpoint: `checkpoints/step_010000.pt`

## Training Results

Loss over training: 2.52 @100 → 1.09 @10,000 steps. Final epoch: train 1.3162, val 1.5110. The model produces English-like character sequences, punctuation, dialogue, and character names.

Example, prompt `ROMEO:`:

```
ROMEO:
What, is it not this?

MERCUTIO:
What, what?

MERCUTIO:
A word of command?

ROMEO:
And so shows the bloody stark to the crown,
And bid her brothers of the gods, and go break him out.
```

## Generation

Supports `temperature`, `top_k`, and `top_p`:

```python
generated = generator.generate(
    prompt="ROMEO:",
    max_new_tokens=200,
    temperature=0.8,
    top_k=20,
    top_p=0.9,
)
```

## Running the Project

```bash
uv sync
uv run python -m scripts.train_smoke_test   # verifies the full pipeline
uv run train.py                             # trains (CUDA if available, else CPU)
uv run python -m scripts.demo_generation    # generates text from the latest checkpoint
```

## Checkpoints

Store model params, optimizer & scheduler state, epoch, and global step, enabling resume after interruption. The trainer auto-loads the latest checkpoint.

## Why Build This From Scratch?

To understand the mechanics — tokenization, embeddings, attention, transformer, language modeling, optimization, training, checkpointing, generation — instead of treating GPT as a black box.

## Next Phase: Coding Language Model

Character-level GPT → BPE/Subword Tokenizer → Clean Code Corpus → Code LM → RoPE → RMSNorm → SwiGLU → Longer Context → Larger Model.

- **Code corpus:** clean source code (Python, TypeScript, JS, Go, Rust, C/C++, Java), filtering `node_modules/`, `.venv/`, `build/`, `dist/`, generated/minified/binary files, vendor dirs, lock files, duplicates.
- **Tokenizer:** BPE subwords (`def calculate_total(items)` as reusable tokens instead of characters).
- **Architecture:** RoPE, RMSNorm, SwiGLU, and longer context for code.
- **Method:** incremental changes with measurable results (Model A: BPE, B: +RoPE, C: +RMSNorm, D: +SwiGLU), using Shakespeare as the baseline.

## License

Primarily an educational implementation. Check licensing for any future training corpus independently.
