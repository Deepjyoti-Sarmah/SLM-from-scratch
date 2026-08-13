#!/usr/bin/env python3

"""Validate the Shakespeare troll SFT dataset before training.

Checks:

    * JSONL is valid and every line has instruction + response.
    * 2780 examples.
    * Character tokenizer keeps vocabulary size 65.
    * Unknown characters = 0.
    * Maximum formatted length <= 120.
    * Examples over 120 = 0.
    * Examples over 128 = 0.
    * No meta/generation instructions inside responses.
    * No accidental C++ (or any digit) characters.

Also reports length statistics for formatted examples and responses.
"""

from __future__ import annotations

import json
import statistics
import sys

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.tokenization.char_tokenizer import CharacterTokenizer

DATASET_PATH = Path("data/shakespeare_troll_sft_large.jsonl")
VOCABULARY_SOURCE = Path("data/tiny_shakespeare.txt")

EXPECTED_EXAMPLES = 2780
MAX_FORMATTED_LENGTH = 120
MODEL_SEQUENCE_LENGTH = 128

META_INSTRUCTION_PATTERNS = [
    "explain this in plain language",
    "in plain language",
    "answer clearly",
    "shakespearean roast",
    "insult the user",
    "include a troll",
    "add a witty",
    "your response should",
    "teach the concept and",
    "in a shakespearean",
    "at the end of your answer",
    "then add",
    "now add",
]


def format_example(
    instruction: str,
    response: str,
) -> str:
    return f"Q: {instruction}\nA: {response}"


def main() -> int:

    failures: list[str] = []

    if not DATASET_PATH.exists():
        print(f"FAIL: missing dataset {DATASET_PATH}")
        failures.append("dataset missing")
        return 1

    rows: list[dict[str, str]] = []

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"FAIL: invalid JSON on line {line_number}: {error}")
                failures.append(f"invalid json line {line_number}")
                continue

            if set(example) != {"instruction", "response"}:
                print(f"FAIL: unexpected keys on line {line_number}: {sorted(example)}")
                failures.append(f"unexpected keys line {line_number}")

            rows.append(
                {
                    "instruction": str(example["instruction"]),
                    "response": str(example["response"]),
                }
            )

    print(f"JSONL valid                      {'OK' if not failures else 'FAIL'}")

    if len(rows) != EXPECTED_EXAMPLES:
        print(
            f"FAIL: expected {EXPECTED_EXAMPLES} examples, got {len(rows)}"
        )
        failures.append("example count")
    else:
        print(f"Examples: {len(rows)}                 OK")

    vocabulary = VOCABULARY_SOURCE.read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer(vocabulary)

    print(f"Vocabulary size: {tokenizer.vocab_size}")

    if tokenizer.vocab_size != 65:
        print("FAIL: vocabulary size is not 65")
        failures.append("vocab size")

    unknown: set[str] = set()
    digits: set[str] = set()

    for row in rows:
        for character in row["instruction"] + row["response"]:
            if character not in tokenizer.token_to_id:
                unknown.add(character)

            if character.isdigit():
                digits.add(character)

    print(f"Unknown characters: {len(unknown)}")

    if unknown:
        print(f"FAIL: unknown characters: {sorted(unknown)}")
        failures.append("unknown characters")

    formatted_lengths = [
        len(format_example(row["instruction"], row["response"]))
        for row in rows
    ]
    response_lengths = [len(row["response"]) for row in rows]

    maximum_formatted = max(formatted_lengths)

    over_120 = sum(1 for length in formatted_lengths if length > MAX_FORMATTED_LENGTH)
    over_128 = sum(
        1 for length in formatted_lengths if length > MODEL_SEQUENCE_LENGTH
    )

    print(
        f"Maximum formatted length: {maximum_formatted}"
        f" ({'OK' if maximum_formatted <= MAX_FORMATTED_LENGTH else 'FAIL'})"
    )

    if maximum_formatted > MAX_FORMATTED_LENGTH:
        failures.append("max formatted length")

    print(f"Examples over 120: {over_120}")

    if over_120:
        failures.append("examples over 120")

    print(f"Examples over 128: {over_128}")

    if over_128:
        failures.append("examples over 128")

    if digits:
        print(f"FAIL: digit characters present: {sorted(digits)}")
        failures.append("numeric vocabulary errors")
    else:
        print("No numeric vocabulary errors      OK")

    joined = "\n".join(
        row["instruction"] + "\n" + row["response"] for row in rows
    ).lower()

    meta_hits = [
        pattern for pattern in META_INSTRUCTION_PATTERNS if pattern in joined
    ]

    if meta_hits:
        print(f"FAIL: meta-instruction patterns found: {sorted(meta_hits)}")
        failures.append("meta-instructions")
    else:
        print("No meta-instructions              OK")

    if "c++" in joined or "c + +" in joined:
        print("FAIL: accidental C++ found")
        failures.append("c++")
    else:
        print("No accidental C++                 OK")

    print()
    print("Formatted example lengths")
    print(f"  Minimum: {min(formatted_lengths)}")
    print(f"  Maximum: {max(formatted_lengths)}")
    print(f"  Mean:    {statistics.mean(formatted_lengths):.2f}")
    print(f"  Median:  {statistics.median(formatted_lengths)}")

    print()
    print("Response lengths")
    print(f"  Minimum: {min(response_lengths)}")
    print(f"  Maximum: {max(response_lengths)}")
    print(f"  Mean:    {statistics.mean(response_lengths):.2f}")

    print()

    if failures:
        print(f"VALIDATION FAILED: {', '.join(sorted(set(failures)))}")
        return 1

    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())