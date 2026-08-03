import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.configs.training_config import TrainingConfig
from src.models.gpt import GPT


class Trainer:
    def __init__(
        self,
        *,
        model: GPT,
        train_dataloader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        config: TrainingConfig,
    ) -> None:
        self.model: GPT = model
        self.train_dataloader: DataLoader = train_dataloader
        self.optimizer: Optimizer = optimizer
        self.scheduler = scheduler
        self.config: TrainingConfig = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        self.current_epoch = 0
        self.global_step = 0

    def train(self) -> None:
        self.model.train()

        for input_ids, target_ids in self.train_dataloader:
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            self.optimizer.zero_grad()

            _, loss = self.model(
                token_ids=input_ids,
                targets=target_ids,
            )

            assert loss is not None

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.gradient_clip,
            )

            self.optimizer.step()

            self.scheduler.step()

            self.global_step += 1
