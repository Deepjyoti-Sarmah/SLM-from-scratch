from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.training.training_checkpoint import TrainingCheckpoint


class CheckpointManager:
    def __init__(
        self,
        *,
        checkpoint_directory: str | Path,
    ) -> None:
        self.checkpoint_directory = Path(checkpoint_directory)

        self.checkpoint_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        *,
        checkpoint: TrainingCheckpoint,
    ) -> Path:
        """
        Save a training checkpoint.

        Returns
        -------
        Path
            Path to the saved checkpoint.
        """

        checkpoint_path = (
            self.checkpoint_directory / f"step_{checkpoint.global_step:06d}.pt"
        )

        torch.save(
            checkpoint.to_dict(),
            checkpoint_path,
        )

        return checkpoint_path

    def load(
        self,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> TrainingCheckpoint:
        """
        Load a training checkpoint.
        """

        checkpoint_dict: dict[str, Any] = torch.load(
            checkpoint_path,
            map_location=map_location,
        )

        return TrainingCheckpoint.from_dict(checkpoint_dict)

    def latest_checkpoint(self) -> Path | None:
        """
        Return the most recent checkpoint.
        """

        checkpoints = sorted(self.checkpoint_directory.glob("step_*.pt"))

        if not checkpoints:
            return None

        return checkpoints[-1]
