# Kaggle SFT Training — Handoff Document

Everything an LLM (or another engineer) needs to run SFT for this project on a Kaggle T4 notebook.

---

## 1. Project Overview

A small GPT-style decoder-only language model built **from scratch in PyTorch** (`~4.78M` parameters). It was pretrained on Tiny Shakespeare characters and produces Shakespearean-style text. We then built an **instruction-following SFT dataset** (coding questions answered with a playful Shakespearean troll) and now want to **fine-tune** the pretrained checkpoint on that dataset.

Current model constraints (do NOT change them):

| Item | Value |
| --- | --- |
| Backbone | GPT decoder-only (from scratch, no HF) |
| Vocabulary size | `65` (character-level tokenizer) |
| Context / sequence length | `128` |
| Embedding dim | `256` |
| Attention heads | `8` |
| Transformer layers | `6` |
| Parameters | ~4,782,336 |
| Pretrained checkpoint | `checkpoints/step_010000.pt` |

---

## 2. What Was Done

### 2.1 Existing state (before this task)

- `data/tiny_shakespeare.txt` — the pretraining corpus (65 characters).
- `checkpoints/step_010000.pt` — the pretrained base model (NOT to be modified).
- `data/shakespeare_troll_sft.jsonl` — a small clean SFT dataset (180 records).
- `data/shakespeare_troll_sft_large.jsonl` — a large SFT dataset (2780 records) that was **broken**:
  - Contained characters **not** in the 65-char vocabulary: `+` (from `C++`), and digits `4`, `5`, `0` (from `HTTP 404`/`HTTP 500`).
  - ~1460 responses contained **generation/meta instructions** (e.g. *"Teach the concept in plain language, with a witty Shakespearean insult at the end."*) that leaked into the target text.
- `src/` — full from-scratch implementation: tokenizer, datasets, GPT model, layers, embeddings, training, inference.
- `scripts/train_sft.py` — SFT training script (reads a JSONL, uses `InstructionDataset`, fine-tunes from the base checkpoint).

### 2.2 What we fixed (dataset cleanup + regeneration)

1. **No generator script existed in the repo**, so we built a new deterministic generator:
   - `scripts/generate_shakespeare_troll_sft.py`
   - All facts, trolls, templates, and war-stories are embedded as static tables inside the script (source of truth).
   - Uses a fixed iteration order → **byte-identical output on every run** (verified via `md5sum`).
2. **Normalized vocabulary-breaking text at the source level:**
   - `C++` → `CPP` (in both questions and facts).
   - `HTTP 500` → `HTTP five zero zero`.
   - `HTTP 404` → `HTTP four zero four`.
3. **Removed all meta/generation instructions** from responses.
   - Response format is now only: `f"{fact} {troll}"`.
4. **Preserved dataset structure and variation:**
   - 132 technical concepts × 10 question templates × 2 trolls = `2640`
   - 7 "war story" questions × 20 trolls = `140`
   - **Total: 2780 records**, matching the original scale.
   - Unique instructions: 1327 (multiplicity `{2: 1320, 20: 7}`).
5. **Backed up the old dataset** to `data/shakespeare_troll_sft_large.backup.jsonl`.
6. **Validated everything:**
   - JSONL structure: 2780 examples, 0 invalid.
   - Tokenizer compatibility: `Vocabulary size: 65`, `Unknown characters: 0`.
   - Meta-instruction grep scan: no matches.
   - No `+` or digit characters remain in the regenerated file.
7. **Training/data-pipeline status:** the base checkpoint, tokenizer, and GPT architecture are untouched. The SFT data pipeline now supports explicit `TOPIC: ...\nQ: ...\nA: ...` conditioning and correct shifted target masking, so loss is computed only on the response and the first response character is not masked.

### 2.3 Example cleaned record

```json
{"instruction": "What is CPP?", "response": "CPP is a compiled systems language offering low level control and powerful abstractions. Now thou knowest the matter; go forth and trouble the compiler no further."}
```

```json
{"instruction": "What is an HTTP five zero zero error?", "response": "An HTTP five zero zero error means the server encountered an internal problem. Now thou knowest the matter; go forth and trouble the compiler no further."}
```

---

## 3. Folder Structure

```
slm-from-scratch/
├── data/
│   ├── tiny_shakespeare.txt                 # pretraining corpus (65 chars)
│   ├── shakespeare_instructions.jsonl       # small general-purpose instruction set
│   ├── shakespeare_troll_sft.jsonl          # small SFT set (180, clean)
│   ├── shakespeare_troll_sft_large.jsonl    # ★ FIXED SFT set (2780, tokens match vocab)
│   └── shakespeare_troll_sft_large.backup.jsonl   # original broken dataset (backup)
│
├── checkpoints/
│   ├── step_010000.pt                       # ★ pretrained base model (SFT starts here)
│   └── sft/                                 # SFT checkpoints will be saved here
│
├── src/
│   ├── configs/
│   │   ├── gpt_config.py                    # GPTConfig (vocab 65, seq 128, dim 256, 8 heads, 6 layers)
│   │   ├── sft_config.py                    # SFTConfig defaults; train_sft.py overrides max_steps to 2000
│   │   └── ...
│   ├── datasets/
│   │   ├── instruction_dataset.py           # ★ SFT dataset: Q/A format, masks prompt, pads to 128
│   │   ├── gpt_dataset.py
│   │   └── ...
│   ├── models/gpt.py                        # GPT model (forward returns logits + loss)
│   ├── tokenization/char_tokenizer.py        # CharacterTokenizer (encode/decode)
│   ├── embeddings/  layers/  inference/  optimizers/  training/
│   └── ...
│
├── scripts/
│   ├── generate_shakespeare_troll_sft.py    # ★ dataset generator (deterministic, source of truth)
│   └── train_sft.py                         # ★ SFT training script
│
├── main.py           # quick pretraining demo entry point
├── train.py          # full pretraining entry point
├── pyproject.toml    # deps: torch, numpy, jupyter, ipykernel
├── uv.lock
├── README.md
└── FIX_SHakespeare_SFT_DATASET.md           # original dataset-fix task spec (ignore for training)
```

Legend: `★` = the files that matter most for the Kaggle SFT run.

---

## 4. What the SFT Training Script Does

`scripts/train_sft.py` (already written — you do NOT need to rewrite it):

1. Loads the pretrained base model from `checkpoints/step_010000.pt` (`strict=True`).
2. Builds a `CharacterTokenizer` from `data/tiny_shakespeare.txt` (vocab = 65).
3. Loads the SFT JSONL via `InstructionDataset`:
   - For TOPIC-conditioned examples, formats each example as `TOPIC: {topic}\nQ: {question}\nA: {response}`.
   - Internally uses prompt `TOPIC: {topic}\nQ: {question}\nA: ` and appends the response.
   - Encodes to character IDs.
   - Allows up to `sequence_length + 1 = 129` IDs, because input and targets are shifted by one for next-token prediction.
   - Raises an error if an example is too long instead of silently truncating a response.
   - Masks only the prompt portion with `-100`; the first non-masked target must be the first response character.
   - Pads with zeros / `-100` to length 128 if shorter.
4. Fine-tunes with `AdamW` (lr `1e-5`, weight decay `0.01`, betas `0.9/0.95`), gradient clipping `1.0`, batch size `8`.
5. Saves SFT checkpoints to `checkpoints/sft/step_{step:06d}.pt` every `100` steps (and at the end).

Important nuance: because this is next-token prediction, targets are shifted by one. The correct prompt mask length is `len(prompt_ids) - 1`, not `len(prompt_ids)`. For the AdamW example, the first supervised target must decode to `A`, not `d` and not `:`. Do not reduce `sequence_length`.

---

## 5. What To Do On Kaggle

### Option A (recommended): upload the repo to a Kaggle Dataset and run a T4 GPU notebook

> **Ready-to-upload artifact:** a clean, validated package already exists.
> Upload **`slm-from-scratch-kaggle-sft.zip`** (fresh, TOPIC-conditioned, TOPIC: `TOPIC: {topic}\nQ: {question}\nA: {response}`).
> It was rebuilt on `2026-08-14` after the TOPIC-conditioning change and verified end-to-end (see Section 6).
> It contains `checkpoints/step_010000.pt`, the 2780-example TOPIC-conditioned dataset, all `src/`, all `scripts/`, `pyproject.toml`, and `uv.lock` — with **no** old SFT checkpoints.

1. **Create a Kaggle Dataset** containing:
   - Upload `slm-from-scratch-kaggle-sft.zip` as-is (it zips the `kaggle_sft_package/` directory; the notebook in step 2 copies it to `/kaggle/working/repo`).
   - Install `uv` or just use `pip` inside the notebook (`pip install torch` is preinstalled on Kaggle; `numpy` also present).
2. **Notebook** (GPU T4 accelerator):

```python
# 1) Pull your Kaggle dataset into the working dir, e.g.
#    The package zip contains a single top-level kaggle_sft_package/ folder,
#    so copy its contents and cd into that folder.
!mkdir -p /kaggle/working
!cp -r /kaggle/input/your-dataset-slug/* /kaggle/working/

import os
os.chdir('/kaggle/working/kaggle_sft_package')
print(os.getcwd())

# 2) Quick dataset validation
!python scripts/validate_sft_dataset.py

# 3) Full SFT readiness check: Q/A formatting, target masking, paths,
#    base checkpoint load, and finite one-batch loss.
!python scripts/verify_sft_ready.py

# 4) Confirm train_sft.py uses the correct paths
!grep -nE 'pretrained_checkpoint|dataset_path|checkpoint_directory|max_steps|checkpoint_every' scripts/train_sft.py

# 5) Run SFT only after all checks pass. This script auto-detects CUDA.
!python scripts/train_sft.py
```

3. **Checkpoints** land in `checkpoints/sft/step_{step:06d}.pt`. With the current `scripts/train_sft.py` settings, `max_steps=2000` and `checkpoint_every=100`, so you get checkpoints every 100 steps through `step_002000.pt`.
4. **Generate / test** on the result, then zip `checkpoints/sft/` and download.

### Option B: adapt the script inline

Copy `scripts/train_sft.py`'s logic into a notebook cell, or change `SFTConfig` values directly in `scripts/train_sft.py` before running. Current training-script settings: `max_steps=2000`, `batch_size=8`, `lr=1e-5`, `checkpoint_every=100`, `sequence_length=128`. Tune `max_steps` / `batch_size` / `lr` on the T4 as needed, but always start from `checkpoints/step_010000.pt`, not from an old SFT checkpoint.

---

## 6. Validation Checklist (dataset was verified BEFORE training)

- [x] Valid JSONL; every line has `instruction` + `response`
- [x] 2780 examples, 0 invalid
- [x] Only the 65 vocabulary characters appear (unknown chars `0`)
- [x] `Vocabulary: 65` unchanged
- [x] No `+` (C++), no digits `4/5/0` (404/500) — uses `CPP`, `HTTP five zero zero`, `HTTP four zero four`
- [x] No generation/meta instructions inside responses
- [x] Technical facts preserved, Shakespearean troll retained
- [x] Base checkpoint `checkpoints/step_010000.pt` untouched
- [x] Tokenizer / GPT architecture untouched
- [x] SFT dataset formatting uses `TOPIC: ...\nQ: ...\nA: ...`
- [x] First non-masked target for AdamW is `A`
- [x] Loss uses `ignore_index=-100`

---

## 7. Important Notes / Gotchas

- **Python version:** `pyproject.toml` requires Python `>=3.14` (for `uv` workflow). Kaggle's default is 3.10/3.11, and the code uses new-style syntax compatible code (`from __future__ import annotations`, slots-only dataclasses) — pure-PyTorch, no exotic deps, so **3.10+ works fine**.
- **Dependencies:** `torch`, `numpy`, `jupyter`, `ipykernel` only. Kaggle ships torch + numpy already; just `!python -m pip install -U ipykernel` if missing.
- **Determinism:** dataset regeneration is deterministic, but SFT training itself still uses a shuffled DataLoader — fine for training, just don't expect fixed seeds.
- **Never** change `vocab_size`, the tokenizer, the GPT config, or `checkpoints/step_010000.pt`. Only the SFT training hyperparameters / `SFTConfig` are meant to be tuned.