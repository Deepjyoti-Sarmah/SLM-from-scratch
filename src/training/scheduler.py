import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from src.configs.training_config import TrainingConfig


def build_scheduler(
    *,
    optimizer: Optimizer,
    config: TrainingConfig,
) -> LambdaLR:
    """
    Creates a learning-rate schedule with:

        1. Linear warmup
        2. Cosine decay
        3. Minimum learning-rate floor
    """

    def lr_lambda(
        current_step: int,
    ) -> float:

        if current_step < config.warmup_steps:
            return float(current_step + 1) / float(config.warmup_steps)

        if current_step >= config.max_steps:
            return config.minimum_learning_rate / config.learning_rate

        decay_progress = current_step - config.warmup_steps / (
            config.max_steps - config.warmup_steps
        )

        cosine_decay = 1.0 + math.cos(math.pi * decay_progress) / 2.0

        minimum_ratio = config.minimum_learning_rate / config.learning_rate

        return minimum_ratio + cosine_decay * (1.0 - minimum_ratio)

    return LambdaLR(
        optimizer=optimizer,
        lr_lambda=lr_lambda,
    )
