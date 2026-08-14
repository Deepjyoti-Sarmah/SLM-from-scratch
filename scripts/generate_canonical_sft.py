#!/usr/bin/env python3

"""Generate a canonical SFT dataset from shakespeare_troll_sft_large.jsonl.

The canonical dataset keeps every unique instruction exactly once and selects
one existing response per instruction using a deterministic prefix-scoring
heuristic. Responses are never rewritten, paraphrased, or merged; the source
dataset is never modified.
"""

from __future__ import annotations

import json
import sys

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

SOURCE_PATH = Path("data/shakespeare_troll_sft_large.jsonl")
OUTPUT_PATH = Path("data/shakespeare_troll_sft_canonical.jsonl")


def extract_topic(
    instruction: str,
) -> str:
    first_line = instruction.split("\n", 1)[0]

    if first_line.startswith("TOPIC: "):
        return first_line[len("TOPIC: ") :].strip()

    return ""


def score_response(
    response: str,
    topic: str,
) -> int:
    """Score a response prefix. Higher is preferred.

    5: begins with the topic (case-insensitive)
    4: begins with 'A ' or 'An '
    3: other direct explanatory responses (any alphabetic start)
    2: begins with 'It '
    1: begins with 'In brief'
    0: anything else
    """

    normalized = response.strip().lower()

    if not normalized:
        return 0

    if topic and normalized.startswith(topic.strip().lower()):
        return 5

    if normalized.startswith(("a ", "an ")):
        return 4

    if normalized.startswith("it "):
        return 2

    if normalized.startswith("in brief"):
        return 1

    if response.strip()[0].isalpha():
        return 3

    return 0


def select_response(
    responses: list[str],
    topic: str,
) -> str:
    """Select one response deterministically.

    Primary key is the prefix score. Within an equal score group we prefer a
    response that begins directly with the topic, then the shorter response,
    then the lexicographically smallest complete response. Randomness, input
    order, and dictionary order never influence the choice.
    """

    scored = [(score_response(response, topic), response) for response in responses]

    best_score = max(score for score, _ in scored)

    candidates = [
        response for score, response in scored if score == best_score
    ]

    def rank(response: str) -> tuple[int, int, str]:
        starts_with_topic = (
            0
            if topic and response.strip().lower().startswith(topic.strip().lower())
            else 1
        )
        return starts_with_topic, len(response), response

    return min(candidates, key=rank)


def load_source(
    path: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            example = json.loads(line)

            if "instruction" not in example or "response" not in example:
                raise ValueError(
                    f"Missing instruction/response keys at line {line_number}"
                )

            rows.append(
                {
                    "instruction": str(example["instruction"]),
                    "response": str(example["response"]),
                }
            )

    return rows


def main() -> int:
    if not SOURCE_PATH.exists():
        print(f"FAIL: missing source dataset {SOURCE_PATH}")
        return 1

    rows = load_source(SOURCE_PATH)

    grouped: dict[str, list[str]] = {}

    for example in rows:
        grouped.setdefault(example["instruction"], []).append(
            example["response"]
        )

    unique_instructions = list(grouped)
    output_rows = []

    for instruction in unique_instructions:
        responses = grouped[instruction]
        topic = extract_topic(instruction)
        output_rows.append(
            {
                "instruction": instruction,
                "response": select_response(responses, topic),
            }
        )

    identical_groups = sum(
        1 for responses in grouped.values() if len(set(responses)) == 1
    )
    conflicting_groups = sum(
        1 for responses in grouped.values() if len(set(responses)) > 1
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for example in output_rows:
            file.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Input rows:                 {len(rows)}")
    print(f"Unique instructions:        {len(unique_instructions)}")
    print(f"Output rows:                {len(output_rows)}")
    print(f"Identical-response groups:  {identical_groups}")
    print(f"Conflicting-response groups:{conflicting_groups}")
    print(f"Wrote canonical dataset:    {OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
