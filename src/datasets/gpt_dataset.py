from __future__ import annotations

import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):
    def __init__(
        self,
        *,
        token_ids: list[int],
        sequence_length: int,
    ) -> None:
        self.token_ids = token_ids
        self.sequence_length = sequence_length

    def __len__(
        self,
    ) -> int:
        return len(self.token_ids) - self.sequence_length

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = torch.tensor(
            self.token_ids[index : index + self.sequence_length],
            dtype=torch.long,
        )

        target_ids = torch.tensor(
            self.token_ids[index + 1 : index + self.sequence_length + 1],
            dtype=torch.long,
        )

        return input_ids, target_ids
