from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.tokenization.char_tokenizer import CharacterTokenizer


class InstructionDataset(Dataset):
    def __init__(
        self,
        *,
        path: str | Path,
        tokenizer: CharacterTokenizer,
        sequence_length: int,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

        self.examples = self._load_examples()

        if not self.examples:
            raise ValueError(f"No instruction examples found in {self.path}")

    def _load_examples(self) -> list[dict[str, str]]:
        examples: list[dict[str, str]] = []

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                example = json.loads(line)

                if "instruction" not in example:
                    raise ValueError(f"Missing 'instruction' at line {line_number}")

                if "response" not in example:
                    raise ValueError(f"Missing 'response' at line {line_number}")

                examples.append(example)

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        example = self.examples[index]

        instruction = example["instruction"]
        response = example["response"]

        prompt = f"Q: {instruction}\nA: "

        full_text = prompt + response

        full_ids = self.tokenizer.encode(full_text)
        prompt_ids = self.tokenizer.encode(prompt)

        # We need sequence_length + 1 tokens because
        # language-model inputs and targets are shifted by one.
        full_ids = full_ids[: self.sequence_length + 1]

        if len(full_ids) < 2:
            raise ValueError(f"Example {index} is too short.")

        input_ids = full_ids[:-1]
        target_ids = full_ids[1:]

        # Do not train on the USER prompt.
        prompt_length = min(
            len(prompt_ids),
            len(target_ids),
        )

        target_ids = target_ids.copy()

        target_ids[:prompt_length] = [-100] * prompt_length

        # Pad remaining positions.
        padding_length = self.sequence_length - len(input_ids)

        if padding_length > 0:
            input_ids.extend([0] * padding_length)

            target_ids.extend([-100] * padding_length)

        return (
            torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            torch.tensor(
                target_ids,
                dtype=torch.long,
            ),
        )
