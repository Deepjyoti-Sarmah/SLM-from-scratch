from pathlib import Path

import torch


class ChecKpointStore:
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
        checkpoint: dict,
        global_step: int,
    ) -> Path:
        """
        Save a training checkpoint.

        Returns
        -------
        Path
            Path to the saved checkpoint.
        """

        checkpoint_path = self.checkpoint_directory / f"step_{global_step:06d}.pt"

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        return checkpoint_path

    def load(
        self,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> dict:
        """
        Load a checkpoint.
        """

        checkpoint_path = Path(checkpoint_path)

        return torch.load(
            checkpoint_path,
            map_location=map_location,
        )

    def latest_checkpoint(self) -> Path | None:
        """
        Return the newest checkpoint.
        """

        checkpoints = sorted(self.checkpoint_directory.glob("step_*.pt"))

        if not checkpoints:
            return None

        return checkpoints[-1]
