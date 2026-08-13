#!/usr/bin/env python3

"""Validate the temporary TOPIC conditioning SFT experiment dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.tokenization.char_tokenizer import CharacterTokenizer

DATASET_PATH = Path("data/sft_conditioning_experiment.jsonl")
VOCABULARY_SOURCE = Path("data/tiny_shakespeare.txt")
EXPECTED_TOPICS = {
    "AdamW",
    "Python",
    "API",
    "database",
    "recursion",
}
MAX_SAFE_LENGTH = 120
MODEL_CONTEXT_LENGTH = 128


def main() -> int:
    tokenizer = CharacterTokenizer(VOCABULARY_SOURCE.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    failures: list[str] = []

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                failures.append(f"invalid JSON line {line_number}: {error}")
                continue

            if set(row) != {"instruction", "response"}:
                failures.append(f"unexpected keys line {line_number}: {sorted(row)}")
                continue

            rows.append(
                {
                    "instruction": str(row["instruction"]),
                    "response": str(row["response"]),
                }
            )

    topics = []
    unknown: set[str] = set()
    lengths: list[int] = []

    for row in rows:
        instruction = row["instruction"]
        response = row["response"]

        if not instruction.startswith("TOPIC: "):
            failures.append(f"instruction missing TOPIC prefix: {instruction!r}")
            continue

        if "\nQ: What is it?" not in instruction:
            failures.append(f"instruction missing question: {instruction!r}")
            continue

        topic = instruction.split("\n", 1)[0].removeprefix("TOPIC: ")
        topics.append(topic)

        formatted = f"{instruction}\nA: {response}"
        lengths.append(len(formatted))
        unknown.update(character for character in formatted if character not in tokenizer.token_to_id)

    if len(rows) != 5:
        failures.append(f"expected 5 rows, got {len(rows)}")

    if set(topics) != EXPECTED_TOPICS:
        failures.append(f"topic mismatch: {sorted(topics)}")

    if unknown:
        failures.append(f"unknown characters: {sorted(unknown)}")

    if lengths and max(lengths) > MAX_SAFE_LENGTH:
        failures.append(f"max length exceeds {MAX_SAFE_LENGTH}: {max(lengths)}")

    if lengths and max(lengths) > MODEL_CONTEXT_LENGTH:
        failures.append(f"max length exceeds {MODEL_CONTEXT_LENGTH}: {max(lengths)}")

    print(f"Rows: {len(rows)}")
    print(f"Topics: {sorted(topics)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Unknown characters: {len(unknown)}")

    if lengths:
        print(f"Minimum formatted length: {min(lengths)}")
        print(f"Maximum formatted length: {max(lengths)}")
        print(f"Examples over 120: {sum(length > MAX_SAFE_LENGTH for length in lengths)}")
        print(f"Examples over 128: {sum(length > MODEL_CONTEXT_LENGTH for length in lengths)}")

    if failures:
        print("VALIDATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
