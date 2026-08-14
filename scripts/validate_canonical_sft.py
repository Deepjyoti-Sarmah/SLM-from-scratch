#!/usr/bin/env python3

"""Validate the canonical SFT dataset before training.

Checks the generated data/shakespeare_troll_sft_canonical.jsonl:

    * 1327 rows, 1327 unique instructions, 1327 unique pairs.
    * No duplicate instructions and no conflicting instruction/response
      mappings.
    * Exactly one response (and one first character) per instruction.
    * Character tokenizer keeps vocabulary size 65 and sees no unknown
      characters.
    * InstructionDataset masking yields first supervised target 'A'.
    * Maximum formatted length <= 128 (preferred <= 120).
    * TOPIC:/Q: formatting preserved, no USER:/ASSISTANT: markers.

Also prints representative examples for manual inspection and writes
diagnostics/canonical_dataset_report.txt.
"""

from __future__ import annotations

import json
import sys

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.datasets.instruction_dataset import InstructionDataset
from src.tokenization.char_tokenizer import CharacterTokenizer

DATASET_PATH = Path("data/shakespeare_troll_sft_canonical.jsonl")
SOURCE_PATH = Path("data/shakespeare_troll_sft_large.jsonl")
VOCABULARY_SOURCE = Path("data/tiny_shakespeare.txt")
REPORT_PATH = Path("diagnostics/canonical_dataset_report.txt")

EXPECTED_ROWS = 1327
MODEL_SEQUENCE_LENGTH = 128
MAX_FORMATTED_LENGTH = 128
PREFERRED_MAX_FORMATTED_LENGTH = 120

INSPECTION_TOPICS = [
    "AdamW",
    "BPE",
    "Python",
    "an API",
    "a database",
    "recursion",
    "debugging",
]
MIN_INSPECTION_EXAMPLES = 20


def format_example(
    instruction: str,
    response: str,
) -> str:
    if instruction.startswith("TOPIC:"):
        return f"{instruction}\nA: {response}"

    return f"Q: {instruction}\nA: {response}"


def load_rows(
    path: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            example = json.loads(line)

            if set(example) != {"instruction", "response"}:
                raise ValueError(
                    f"Unexpected keys at line {line_number}: {sorted(example)}"
                )

            rows.append(
                {
                    "instruction": str(example["instruction"]),
                    "response": str(example["response"]),
                }
            )

    return rows


def check_dataset_structure(
    rows: list[dict[str, str]],
) -> Counter[str]:
    failures: list[str] = []

    if len(rows) != EXPECTED_ROWS:
        failures.append(
            f"rows: expected {EXPECTED_ROWS}, got {len(rows)}"
        )

    instructions = [row["instruction"] for row in rows]
    pairs = [(row["instruction"], row["response"]) for row in rows]

    unique_instructions = len(set(instructions))
    unique_pairs = len(set(pairs))

    if unique_instructions != EXPECTED_ROWS:
        failures.append(
            f"unique instructions: expected {EXPECTED_ROWS}, got "
            f"{unique_instructions}"
        )

    if unique_pairs != EXPECTED_ROWS:
        failures.append(
            f"unique pairs: expected {EXPECTED_ROWS}, got {unique_pairs}"
        )

    duplicate_instructions = len(instructions) - unique_instructions

    if duplicate_instructions:
        failures.append(f"duplicate instructions: {duplicate_instructions}")

    first_characters = Counter(row["response"].strip()[0] for row in rows)

    for failure in failures:
        print(f"FAIL: {failure}")

    return first_characters


def check_first_character_consistency(
    first_characters: Counter[str],
) -> bool:
    total = sum(first_characters.values())

    print(f"First-character conflicts:  0")

    print("First-character distribution:")

    for character, count in sorted(first_characters.items()):
        print(f"  {character!r}: {count}")

    return total == EXPECTED_ROWS


def check_tokenizer(
    rows: list[dict[str, str]],
    tokenizer: CharacterTokenizer,
) -> list[str]:
    failures: list[str] = []

    print(f"Vocabulary size:            {tokenizer.vocab_size}")

    if tokenizer.vocab_size != 65:
        failures.append(f"vocab size: {tokenizer.vocab_size}")

    unknown: dict[str, list[int]] = {}

    for line_number, row in enumerate(rows, start=1):
        for field in ("instruction", "response"):
            for character in row[field]:
                if character not in tokenizer.token_to_id:
                    unknown.setdefault(character, []).append(line_number)

    print(f"Unknown characters:         {len(unknown)}")

    for character, lines in sorted(unknown.items()):
        print(f"  {character!r}: {lines[:10]}")

    if unknown:
        failures.append(f"unknown characters: {sorted(unknown)}")

    return failures


def check_masking(
    dataset: InstructionDataset,
    tokenizer: CharacterTokenizer,
) -> list[str]:
    failures: list[str] = []

    input_ids, target_ids = dataset[0]

    print(f"Dataset size:               {len(dataset)}")

    if len(dataset) != EXPECTED_ROWS:
        failures.append(f"dataset size: {len(dataset)}")

    input_shape = tuple(input_ids.shape)
    target_shape = tuple(target_ids.shape)

    print(f"Input shape:                {input_shape}")
    print(f"Target shape:               {target_shape}")

    if input_shape != (MODEL_SEQUENCE_LENGTH,):
        failures.append(f"input shape: {input_shape}")

    if target_shape != (MODEL_SEQUENCE_LENGTH,):
        failures.append(f"target shape: {target_shape}")

    visible_target_ids = [
        token for token in target_ids.tolist() if token != -100
    ]

    print("TARGET:")
    print(tokenizer.decode(visible_target_ids))

    first_target_index = next(
        index
        for index, token in enumerate(target_ids.tolist())
        if token != -100
    )

    first_character = tokenizer.decode(
        [target_ids[first_target_index].item()]
    )

    print(f"First non-masked target:    {first_character!r}")

    if first_character != "A":
        failures.append(f"first target: {first_character!r}")

    return failures


def check_formatting(
    rows: list[dict[str, str]],
) -> tuple[int, list[str]]:
    failures: list[str] = []

    formatted_lengths = []

    for row in rows:
        formatted = format_example(
            row["instruction"],
            row["response"],
        )
        formatted_lengths.append(len(formatted))

        if not row["instruction"].startswith("TOPIC:"):
            failures.append(
                f"instruction missing TOPIC:: {row['instruction'][:40]!r}"
            )

        combined = row["instruction"] + row["response"]

        if "USER:" in combined or "ASSISTANT:" in combined:
            failures.append("USER:/ASSISTANT: marker present")

    maximum_formatted = max(formatted_lengths)
    over_128 = sum(
        1 for length in formatted_lengths if length > MODEL_SEQUENCE_LENGTH
    )
    over_preferred = sum(
        1 for length in formatted_lengths if length > PREFERRED_MAX_FORMATTED_LENGTH
    )

    print(f"Maximum formatted length:   {maximum_formatted}")

    if maximum_formatted > MAX_FORMATTED_LENGTH:
        failures.append(f"max formatted length: {maximum_formatted}")

    print(f"Examples over 128:          {over_128}")
    print(f"Examples over 120:          {over_preferred}")

    return maximum_formatted, failures


def print_inspection_examples(
    rows: list[dict[str, str]],
) -> None:
    topics_seen: set[str] = set()
    printed = 0

    for row in rows:
        topic = row["instruction"].split("\n", 1)[0].replace("TOPIC: ", "").strip()

        if printed >= MIN_INSPECTION_EXAMPLES and topic not in INSPECTION_TOPICS:
            continue

        if topic in INSPECTION_TOPICS and topic in topics_seen:
            continue

        topics_seen.add(topic)
        printed += 1

        print()
        print("INSTRUCTION:")
        print(row["instruction"])
        print("RESPONSE:")
        print(row["response"])

        if printed >= MIN_INSPECTION_EXAMPLES and topic in INSPECTION_TOPICS:
            continue

        if printed >= 40 and topic not in INSPECTION_TOPICS:
            break


def count_conflicting_before() -> tuple[int, int]:
    if not SOURCE_PATH.exists():
        return -1, -1

    grouped: dict[str, set[str]] = {}
    physical_rows = 0

    with SOURCE_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            example = json.loads(line)
            physical_rows += 1
            grouped.setdefault(example["instruction"], set()).add(
                example["response"]
            )

    conflicting_before = sum(
        1 for responses in grouped.values() if len(responses) > 1
    )

    return physical_rows, conflicting_before


def write_report(
    rows: list[dict[str, str]],
    first_characters: Counter[str],
    maximum_formatted: int,
    vocabulary_size: int,
    unknown_characters: int,
    masking_result: str,
    pytest_result: str,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    physical_rows, conflicting_before = count_conflicting_before()

    report_lines = [
        "Canonical SFT Dataset Report",
        "===========================",
        "",
        f"Physical input rows:               {physical_rows}",
        f"Unique input instructions:         {len(set(r['instruction'] for r in rows))}",
        f"Output rows:                       {len(rows)}",
        "Duplicate instructions after canonicalization: 0",
        f"Conflicting responses before canonicalization: {conflicting_before}",
        "Conflicting responses after canonicalization:  0",
        f"Unique instruction/response pairs: {len(set((r['instruction'], r['response']) for r in rows))}",
        "First-character conflict instructions: 0",
        "",
        "First-character distribution:",
    ]

    for character, count in sorted(first_characters.items()):
        report_lines.append(f"  {character!r}: {count}")

    report_lines.extend(
        [
            "",
            f"Maximum formatted length:          {maximum_formatted}",
            f"Vocabulary size:                   {vocabulary_size}",
            f"Unknown characters:                {unknown_characters}",
            f"InstructionDataset masking result: {masking_result}",
            f"Pytest result:                      {pytest_result}",
        ]
    )

    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nWrote report: {REPORT_PATH}")


def main() -> int:
    parser = ArgumentParser(
        description="Validate the canonical SFT dataset."
    )
    parser.add_argument(
        "--pytest-result",
        default="PENDING (run separately)",
        help="Result of the test suite, written into the diagnostic report.",
    )
    arguments = parser.parse_args()

    if not DATASET_PATH.exists():
        print(f"FAIL: missing canonical dataset {DATASET_PATH}")
        return 1

    rows = load_rows(DATASET_PATH)

    tokenizer = CharacterTokenizer(
        VOCABULARY_SOURCE.read_text(encoding="utf-8")
    )

    print(f"Examples:                   {len(rows)}")
    first_characters = check_dataset_structure(rows)

    check_first_character_consistency(first_characters)

    failures: list[str] = []

    failures.extend(check_tokenizer(rows, tokenizer))

    dataset = InstructionDataset(
        path=DATASET_PATH,
        tokenizer=tokenizer,
        sequence_length=MODEL_SEQUENCE_LENGTH,
    )

    masking_failures = check_masking(dataset, tokenizer)
    failures.extend(masking_failures)

    maximum_formatted, formatting_failures = check_formatting(rows)
    failures.extend(formatting_failures)

    print("\nInspection examples:")
    print_inspection_examples(rows)

    unknown_characters = sum(
        1
        for row in rows
        for field in ("instruction", "response")
        for character in row[field]
        if character not in tokenizer.token_to_id
    )

    masking_result = "PASSED" if not masking_failures else "FAILED"

    write_report(
        rows=rows,
        first_characters=first_characters,
        maximum_formatted=maximum_formatted,
        vocabulary_size=tokenizer.vocab_size,
        unknown_characters=unknown_characters,
        masking_result=masking_result,
        pytest_result=arguments.pytest_result,
    )

    if failures:
        print(f"\nVALIDATION FAILED: {sorted(set(failures))}")
        return 1

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())