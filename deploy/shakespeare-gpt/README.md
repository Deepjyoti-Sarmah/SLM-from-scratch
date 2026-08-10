# Shakespeare GPT

A small, self-contained character-level GPT trained on Tiny Shakespeare.

- **Parameters:** 4.78M
- **Tokenizer:** character-level (65 tokens)
- **Training corpus:** Tiny Shakespeare
- **Context length:** 128 tokens
- **Training steps:** 10,000

## Architecture

- 6 transformer blocks
- 256-dimensional embeddings
- 8 attention heads
- Learned token + positional embeddings
- Pre-LayerNorm residual transformer with a fused QKV projection and a
  4x MLP expansion
- Weight-tied LM head and token embedding

This is a plain PyTorch model. It does not use the `transformers`,
`accelerate`, `peft`, or `trl` libraries.

## Quick start

```bash
git clone <model-repository>
cd shakespeare-GPT
pip install -r requirements.txt
python inference.py --prompt "ROMEO:"
```

### Install

```bash
pip install -r requirements.txt
```

Requirements:

- `torch`
- `safetensors`
- `huggingface_hub`

## Usage

Run inference with a prompt (the model continues the text):

```bash
python inference.py --prompt "ROMEO:"
```

Optional sampling flags:

```bash
python inference.py \
    --prompt "ROMEO:" \
    --max-new-tokens 200 \
    --temperature 0.8 \
    --top-k 40
```

Supported generation parameters:

- `--max-new-tokens` — number of tokens to generate
- `--temperature` — sampling temperature (`0.0` = greedy)
- `--top-k` — top-k sampling filter
- `--top-p` — nucleus sampling filter
- `--seed` — seed for reproducible sampling

### CPU

```bash
python inference.py \
    --device cpu \
    --prompt "ROMEO:"
```

The model runs on CPU with no special setup. It is slower than GPU but
works on any machine.

### GPU (CUDA)

```bash
python inference.py \
    --device cuda \
    --prompt "ROMEO:"
```

CUDA is used automatically when available; the model never requires it.

### Download from the Hub

The weights are stored as a Git LFS object. If you `git clone` this repo,
install git-lfs first or you will only get small LFS *pointer* files:

```bash
git lfs install
git clone https://huggingface.co/Deepjyoti/shakespeare-GPT
cd shakespeare-GPT
pip install -r requirements.txt
python inference.py --prompt "ROMEO:"
```

Alternatively, download the files with the `hf` / `huggingface_hub` tooling
(no git-lfs needed):

```bash
pip install -r requirements.txt

# hf CLI
hf download Deepjyoti/shakespeare-GPT --local-dir .

# or Python
python -c "from huggingface_hub import snapshot_download; snapshot_download('Deepjyoti/shakespeare-GPT', local_dir='.')"
```

You can also let `inference.py` fetch the weights automatically:

```bash
python inference.py \
    --model-id Deepjyoti/shakespeare-GPT \
    --prompt "ROMEO:"
```

A convenience script that downloads everything into a local cache and
runs inference:

```bash
python download_and_run.py \
    --model-id Deepjyoti/shakespeare-GPT \
    --prompt "ROMEO:"
```

## Repository layout

```
README.md
config.json          # model hyper-parameters
metadata.json        # training metadata
model.safetensors     # inference weights
model.py              # standalone model implementation
tokenizer.py          # tokenizer implementation
tokenizer.json        # token-to-id mapping
inference.py          # inference entry point
download_and_run.py   # download-and-infer convenience script
requirements.txt
tests/                # tokenizer, loading, generation, and CPU tests
```

## Tests

```bash
pip install pytest
python -m pytest tests -q
```

## Limitations

This is an experimental research model. Please keep expectations
appropriate:

- It is a **small GPT** (4.78M parameters) trained on ~100K lines of
  Shakespeare for 10,000 steps.
- It is **not** a general-purpose language model.
- It is **not** a coding model and cannot write or reason about code.
- Output is often grammatically broken or nonsensical, especially beyond
  the first few sentences.
- It is primarily a demonstration of a from-scratch GPT — useful for
  learning and experimentation, not production use.

Do not rely on it for factual or creative writing.