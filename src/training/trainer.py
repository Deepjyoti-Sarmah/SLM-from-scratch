import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.configs.training_config import TrainingConfig
from src.models.gpt import GPT
from src.training.checkpoint import CheckpointManager
from src.training.training_checkpoint import TrainingCheckpoint


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
        checkpoint_manager: CheckpointManager,
    ) -> None:
        self.model: GPT = model
        self.train_dataloader: DataLoader = train_dataloader
        self.validation_dataloader: DataLoader = validation_dataloader
        self.optimizer: Optimizer = optimizer
        self.scheduler = scheduler
        self.config: TrainingConfig = config
        self.checkpoint_manager = checkpoint_manager

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        self.current_epoch = 0
        self.global_step = 0

    def train(self) -> None:
        """
        Train the model for multiple epochs
        """

        self._resume_from_checkpoint()

        if self.current_epoch >= self.config.num_epochs:
            print("Trainigng is already complete.")
            return

        self.model.train()

        for epoch in range(
            self.config.num_epochs,
            self.config.num_epochs,
        ):
            self.current_epoch = epoch + 1

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
                    self._log_training_step(batch_loss)

            average_train_loss = epoch_loss / num_batches

            validation_loss = self._validate()

            self._log_epoch_summary(
                train_loss=average_train_loss,
                validation_loss=validation_loss,
            )

            self._save_checkpoint()

    def _resume_from_checkpoint(self) -> None:
        """
        Resume training from the latest checkpoint if one exists.
        """

        checkpoint_path = self.checkpoint_manager.latest_checkpoint()

        if checkpoint_path is None:
            print("No checkpoint found. Starting fresh.")
            return

        checkpoint = self.checkpoint_manager.load(
            checkpoint_path=checkpoint_path,
            map_location=self.device,
        )

        self._load_checkpoint(checkpoint)

        print(f"Resumed training from {checkpoint_path}")

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

        input_ids, target_ids = self._move_batch_to_device(
            input_ids=input_ids,
            target_ids=target_ids,
        )

        self.optimizer.zero_grad(set_to_none=True)

        loss = self._compute_loss(
            input_ids=input_ids,
            target_ids=target_ids,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.gradient_clip,
        )

        self.optimizer.step()

        self.scheduler.step()

        self.global_step += 1

        if self.global_step % self.config.checkpoint_every == 0:
            self._save_checkpoint()

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
                input_ids, target_ids = self._move_batch_to_device(
                    input_ids=input_ids,
                    target_ids=target_ids,
                )

                loss = self._compute_loss(
                    input_ids=input_ids,
                    target_ids=target_ids,
                )

                total_loss += loss.item()
                num_batches += 1

        self.model.train()

        if num_batches == 0:
            raise RuntimeError("Validation dataloader is empty")

        return total_loss / num_batches

    def _compute_loss(
        self,
        *,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run a forward pass and compute the language modeling loss.
        """

        _, loss = self.model(
            token_ids=input_ids,
            targets=target_ids,
        )

        assert loss is not None

        return loss

    def _move_batch_to_device(
        self,
        *,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Move a batch to the training device.
        """

        return (
            input_ids.to(self.device),
            target_ids.to(self.device),
        )

    def _load_checkpoint(
        self,
        checkpoint: TrainingCheckpoint,
    ) -> None:
        """
        Restore the training state.
        """

        self.model.load_state_dict(checkpoint.model_state)
        self.optimizer.load_state_dict(checkpoint.optimizer_state)
        self.scheduler.load_state_dict(checkpoint.scheduler_state)

        self.current_epoch = checkpoint.epoch
        self.global_step = checkpoint.global_step

    def _create_checkpoint(self) -> TrainingCheckpoint:
        """
        Capture the current training state.
        """

        return TrainingCheckpoint.from_training_state(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            global_step=self.global_step,
        )

    def _save_checkpoint(
        self,
    ) -> None:
        """
        Save the current training state.
        """

        checkpoint = self._create_checkpoint()

        checkpoint_path = self.checkpoint_manager.save(
            checkpoint=checkpoint,
        )

        print(f"Checkpoint saved to {checkpoint_path}")

    def _log_training_step(
        self,
        loss: float,
    ) -> None:
        """
        Log one optimization step.
        """

        learning_rate = self.optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {self.current_epoch:03d}/{self.config.num_epochs:03d}] "
            f"[Step {self.global_step:06d}] "
            f"Loss: {loss:.4f} "
            f"LR: {learning_rate:.6e}"
        )

    def _log_epoch_summary(
        self,
        *,
        train_loss: float,
        validation_loss: float,
    ) -> None:
        """
        Log epoch metrics.
        """

        print(
            f"Epoch {self.current_epoch:03d}/{self.config.num_epochs:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {validation_loss:.4f}"
        )
