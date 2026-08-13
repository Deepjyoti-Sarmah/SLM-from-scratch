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

    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be greater than or equal to 0")

    if config.max_steps <= 0:
        raise ValueError("max_steps must be greater than 0")

    if config.warmup_steps >= config.max_steps:
        raise ValueError("warmup_steps must be smaller than max_steps")

    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")

    if config.minimum_learning_rate < 0:
        raise ValueError("minimum_learning_rate must be greater than or equal to 0")

    if config.minimum_learning_rate > config.learning_rate:
        raise ValueError("minimum_learning_rate must not be greater than learning_rate")

    minimum_ratio = config.minimum_learning_rate / config.learning_rate

    def lr_lambda(
        current_step: int,
    ) -> float:
        """
        Return the learning-rate multiplier for the current step.

        The returned value is multiplied by the optimizer's
        base learning rate.
        """

        # warmpup
        if current_step < config.warmup_steps:
            return float(current_step + 1) / float(config.warmup_steps)

        # after training
        if current_step >= config.max_steps:
            return config.minimum_learning_rate / config.learning_rate

        decay_steps = config.max_steps - config.warmup_steps

        decay_progress = (current_step - config.warmup_steps) / (decay_steps)

        cosine_decay = (1.0 + math.cos(math.pi * decay_progress)) / 2.0

        return minimum_ratio + cosine_decay * (1.0 - minimum_ratio)

    return LambdaLR(
        optimizer=optimizer,
        lr_lambda=lr_lambda,
    )
