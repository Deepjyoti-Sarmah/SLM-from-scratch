from dataclasses import dataclass


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 64

    learning_rate: float = 3e-4

    weight_decay: float = 0.1

    beta1: float = 0.9

    beta2: float = 0.95

    gradient_clip: float = 1.0

    num_epochs: int = 10

    log_every: int = 100

    checkpoint_every: int = 1000

    warmup_steps: int = 200

    max_steps: int = 10_000

    minimum_learning_rate: float = 3e-5
