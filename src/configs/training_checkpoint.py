from dataclasses import dataclass

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from src.models.gpt import GPT


@dataclass(slots=True)
class TrainingCheckpoint:
    model_state: dict
    optimizer_state: dict
    scheduler_state: dict

    epoch: int
    global_step: int

    def to_dict(self) -> dict:
        return {
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "epoch": self.epoch,
            "global_step": self.global_step,
        }

    @classmethod
    def from_training_state(
        cls,
        *,
        model: GPT,
        optimizer:Optimizer,
        scheduler : LRScheduler,
        epoch: int,
        global_step: int,
    ) -> "TrainingCheckpoint":
