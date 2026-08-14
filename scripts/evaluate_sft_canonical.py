#!/usr/bin/env python3

"""Evaluate a checkpoint against the canonical SFT evaluation set.

Measures, on a fixed deterministic set of prompts taken from the canonical
dataset:

    * masked loss
    * first-character accuracy
    * full token accuracy
    * topic correctness (generated response begins with the topic)
    * answer quality (generated response length and body overlap)
    * behavior across question phrasings (greedy generation)

Usage:

    python scripts/evaluate_sft_canonical.py \
        --checkpoint checkpoints/step_010000.pt
    python scripts/evaluate_sft_canonical.py \
        --checkpoint checkpoints/sft_canonical/step_000500.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from collections import Counter, defaultdict
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.configs.gpt_config import GPTConfig
from src.models.gpt import GPT
from src.tokenization.char_tokenizer import CharacterTokenizer

DATASET_PATH = Path("data/shakespeare_troll_sft_canonical.jsonl")
VOCABULARY_SOURCE = Path("data/tiny_shakespeare.txt")
MODEL_SEQUENCE_LENGTH = 128
EVAL_EXAMPLES = 100
MAX_NEW_TOKENS = 96


def extract_topic(
    instruction: str,
) -> str:
    first_line = instruction.split("\n", 1)[0]

    if first_line.startswith("TOPIC: "):
        return first_line[len("TOPIC: ") :].strip()

    return ""


def question_type(
    instruction: str,
) -> str:
    for line in instruction.split("\n"):
        if line.startswith("Q: "):
            return line[3:].strip().rstrip("?")

    return "unknown"


def build_eval_set(
    tokenizer: CharacterTokenizer,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            example = json.loads(line)
            rows.append(
                {
                    "instruction": str(example["instruction"]),
                    "response": str(example["response"]),
                }
            )

    random.Random(42).shuffle(rows)

    eval_rows = rows[:EVAL_EXAMPLES]
    eval_rows.sort(key=lambda row: row["instruction"])

    return eval_rows


def load_model(
    checkpoint_path: str,
    tokenizer: CharacterTokenizer,
    device: torch.device,
) -> GPT:
    model = GPT(
        config=GPTConfig(
            vocab_size=tokenizer.vocab_size,
            max_sequence_length=MODEL_SEQUENCE_LENGTH,
            embedding_dim=256,
            num_heads=8,
            num_layers=6,
        )
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )

    model.to(device)
    model.eval()

    return model


def compute_loss(
    model: GPT,
    tokenizer: CharacterTokenizer,
    rows: list[dict[str, str]],
    device: torch.device,
) -> float:
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for row in rows:
            instruction = row["instruction"]
            response = row["response"]

            prompt = f"{instruction}\nA: "
            full_text = prompt + response

            full_ids = tokenizer.encode(full_text)[: MODEL_SEQUENCE_LENGTH + 1]
            prompt_ids = tokenizer.encode(prompt)

            if len(full_ids) < 2:
                continue

            input_ids = torch.tensor(
                full_ids[:-1],
                dtype=torch.long,
            ).unsqueeze(0)

            target_ids = torch.tensor(
                full_ids[1:],
                dtype=torch.long,
            )

            mask_length = min(
                max(len(prompt_ids) - 1, 0),
                len(target_ids),
            )

            target_ids[:mask_length] = -100

            input_ids = input_ids.to(device)
            target_ids = target_ids.unsqueeze(0).to(device)

            _, loss = model(
                token_ids=input_ids,
                targets=target_ids,
            )

            assert loss is not None
            total_loss += loss.item()
            count += 1

    return total_loss / max(count, 1)


@torch.no_grad()
def generate_response(
    model: GPT,
    tokenizer: CharacterTokenizer,
    prompt: str,
    device: torch.device,
) -> str:
    token_ids = torch.tensor(
        tokenizer.encode(prompt),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    generated = model.generate(
        token_ids=token_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
    )

    generated_text = tokenizer.decode(generated[0].tolist())

    return generated_text[len(prompt) :]


def evaluate(
    checkpoint_path: str,
) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CharacterTokenizer(
        VOCABULARY_SOURCE.read_text(encoding="utf-8")
    )

    model = load_model(checkpoint_path, tokenizer, device)

    eval_rows = build_eval_set(tokenizer)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Eval examples: {len(eval_rows)}")
    print()

    loss = compute_loss(model, tokenizer, eval_rows, device)
    print(f"Masked eval loss: {loss:.4f}")

    total = len(eval_rows)
    first_char_hits = 0
    token_hits = 0
    token_total = 0
    topic_hits = 0
    answer_hits = 0
    response_lengths = []

    phrasing_stats: dict[str, Counter[str]] = defaultdict(lambda: Counter())

    generation_samples = []

    for row in eval_rows:
        instruction = row["instruction"]
        reference = row["response"]
        topic = extract_topic(instruction)
        phrasing = question_type(instruction)

        prompt = f"{instruction}\nA: "

        generated = generate_response(
            model,
            tokenizer,
            prompt,
            device,
        ).rstrip()

        if generated:
            first_char_hits += generated[0] == reference[0]

        reference_tokens = list(reference)
        generated_tokens = list(generated)

        for generated_token, reference_token in zip(
            generated_tokens,
            reference_tokens,
        ):
            token_total += 1
            token_hits += generated_token == reference_token

        response_lengths.append(len(generated_tokens))

        topic_ok = bool(topic) and generated.lower().startswith(topic.lower())
        topic_hits += topic_ok

        body_match = generated.startswith(reference.split(".", 1)[0])
        answer_hits += body_match

        phrasing_stats[phrasing]["first_char"] += generated[0] == reference[0] if generated else 0
        phrasing_stats[phrasing]["topic"] += topic_ok
        phrasing_stats[phrasing]["count"] += 1

        if len(generation_samples) < 12:
            generation_samples.append(
                {
                    "instruction": instruction,
                    "reference": reference,
                    "generated": generated,
                }
            )

    print(f"First-character accuracy:  {first_char_hits / total:.4f}")
    print(f"Full token accuracy:       {token_hits / max(token_total, 1):.4f}")
    print(f"Topic correctness:         {topic_hits / total:.4f}")
    print(f"Answer body overlap:       {answer_hits / total:.4f}")
    print(f"Mean generated length:     {sum(response_lengths) / total:.1f}")

    print()
    print("Question phrasing breakdown (first-character / topic):")
    for phrasing in sorted(phrasing_stats):
        stats = phrasing_stats[phrasing]
        count = stats["count"]
        print(
            f"  {phrasing!r}: {count} examples, "
            f"first-char {stats['first_char'] / count:.2f}, "
            f"topic {stats['topic'] / count:.2f}"
        )

    print()
    print("Greedy generation samples:")
    for sample in generation_samples:
        print()
        print("INSTRUCTION:")
        print(sample["instruction"])
        print("REFERENCE:")
        print(sample["reference"])
        print("GENERATED:")
        print(sample["generated"])

    return {
        "checkpoint": checkpoint_path,
        "loss": loss,
        "first_char_accuracy": first_char_hits / total,
        "token_accuracy": token_hits / max(token_total, 1),
        "topic_correctness": topic_hits / total,
        "answer_overlap": answer_hits / total,
        "mean_generated_length": sum(response_lengths) / total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the canonical SFT eval set."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a .pt checkpoint to evaluate.",
    )
    arguments = parser.parse_args()

    evaluate(arguments.checkpoint)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())