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
        validation_dataloader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        config: TrainingConfig,
    ) -> None:
        self.model: GPT = model
        self.train_dataloader: DataLoader = train_dataloader
        self.validation_dataloader: DataLoader = validation_dataloader
        self.optimizer: Optimizer = optimizer
        self.scheduler = scheduler
        self.config: TrainingConfig = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        self.current_epoch = 0
        self.global_step = 0

    def train(self) -> None:
        """
        Train the model for multiple epochs
        """

        self.model.train()

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            epoch_loss = 0.0
            num_batches = 0

            for input_ids, target_ids in self.train_dataloader:
                batch_loss = self._train_step(
                    input_ids=input_ids,
                    target_ids=target_ids,
                )

                epoch_loss += batch_loss
                num_batches += 1

                if self.global_step % self.config.log_every == 0:
                    learning_rate = self.optimizer.param_groups[0]["lr"]

                    print(
                        f"[Epoch {epoch + 1:>2}/{self.config.num_epochs}] "
                        f"[Step {self.global_step:>6}] "
                        f"Loss: {batch_loss:.4f} "
                        f"LR: {learning_rate:.6f}"
                    )

            average_loss = epoch_loss / num_batches

            validation_loss = self._validate()

            print(
                f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                f"Train Loss: {average_loss:.4f} | "
                f"Validation Loss: {validation_loss:.4f}"
            )

    def _train_step(
        self,
        *,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> float:
        """
        Perform one optimization step.

        Returns
        -------
        float
            Batch loss.
        """

        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

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

        return loss.item()

    def _validate(self) -> float:
        """
        Evaluate the model on the validation dataset.

        Returns
        -------
        float
            Average validation loss.
        """

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for input_ids, target_ids in self.validation_dataloader:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                _, loss = self.model(
                    token_ids=input_ids,
                    targets=target_ids,
                )

                assert loss is not None

                total_loss += loss.item()
                num_batches += 1

        self.model.train()

        return total_loss / num_batches
