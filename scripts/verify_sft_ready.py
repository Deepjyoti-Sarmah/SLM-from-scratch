#!/usr/bin/env python3

"""Verify the repository is ready for corrected SFT training.

This script is intended to be run before `scripts/train_sft.py`, locally or on
Kaggle. It checks dataset size/formatting, target masking, SFT paths, and a
single forward loss from the base checkpoint.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.configs.gpt_config import GPTConfig
from src.datasets.instruction_dataset import InstructionDataset
from src.models.gpt import GPT
from src.tokenization.char_tokenizer import CharacterTokenizer

DATASET_PATH = Path("data/shakespeare_troll_sft_large.jsonl")
TOKENIZER_SOURCE = Path("data/tiny_shakespeare.txt")
BASE_CHECKPOINT = Path("checkpoints/step_010000.pt")
TRAIN_SCRIPT = Path("scripts/train_sft.py")
EXPECTED_EXAMPLES = 2780
SEQUENCE_LENGTH = 128


def _assert_train_script_paths() -> None:
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")

    required = [
        'pretrained_checkpoint=("checkpoints/step_010000.pt")',
        'dataset_path=("data/shakespeare_troll_sft_large.jsonl")',
        'checkpoint_directory=("checkpoints/sft")',
    ]

    for text in required:
        if text not in source:
            raise AssertionError(f"Missing expected train_sft.py setting: {text}")

    forbidden_patterns = [
        "data/shakespeare_troll_sft.jsonl",
        "USER:",
        "ASSISTANT:",
        "checkpoints/sft/step_",
    ]

    for pattern in forbidden_patterns:
        if pattern in source:
            raise AssertionError(f"Forbidden train_sft.py text found: {pattern}")

    match = re.search(r"max_steps=(\d+)", source)
    if match:
        print(f"train_sft max_steps: {match.group(1)}")


def main() -> int:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    if not BASE_CHECKPOINT.exists():
        raise FileNotFoundError(BASE_CHECKPOINT)

    tokenizer = CharacterTokenizer(TOKENIZER_SOURCE.read_text(encoding="utf-8"))
    print("Vocabulary size:", tokenizer.vocab_size)
    assert tokenizer.vocab_size == 65

    dataset = InstructionDataset(
        path=DATASET_PATH,
        tokenizer=tokenizer,
        sequence_length=SEQUENCE_LENGTH,
    )

    print("Dataset size:", len(dataset))
    assert len(dataset) == EXPECTED_EXAMPLES

    input_ids, target_ids = dataset[0]
    visible_targets = [token for token in target_ids.tolist() if token != -100]
    first_target_index = next(
        index for index, token in enumerate(target_ids.tolist()) if token != -100
    )
    first_character = tokenizer.decode([target_ids[first_target_index].item()])

    print("Input shape:", tuple(input_ids.shape))
    print("Target shape:", tuple(target_ids.shape))
    input_text = tokenizer.decode(input_ids.tolist())

    print("Input starts:", repr(input_text[:30]))
    print("Target starts:", repr(tokenizer.decode(visible_targets)[:30]))
    print("First non-masked target:", repr(first_character))

    assert tuple(input_ids.shape) == (SEQUENCE_LENGTH,)
    assert tuple(target_ids.shape) == (SEQUENCE_LENGTH,)
    assert input_text.startswith(("Q: ", "TOPIC: "))
    assert "\nA: " in input_text
    assert first_character == "A"

    _assert_train_script_paths()

    model = GPT(
        config=GPTConfig(
            vocab_size=tokenizer.vocab_size,
            max_sequence_length=SEQUENCE_LENGTH,
            embedding_dim=256,
            num_heads=8,
            num_layers=6,
        )
    )

    checkpoint = torch.load(
        BASE_CHECKPOINT,
        map_location="cpu",
    )
    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )
    model.eval()

    with torch.no_grad():
        logits, loss = model(
            token_ids=input_ids.unsqueeze(0),
            targets=target_ids.unsqueeze(0),
        )

    assert loss is not None
    print("Logits shape:", tuple(logits.shape))
    print("Loss:", loss.item())
    assert logits.shape == (1, SEQUENCE_LENGTH, tokenizer.vocab_size)
    assert math.isfinite(loss.item())

    print("SFT READY: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
