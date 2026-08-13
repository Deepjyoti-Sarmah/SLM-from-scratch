from dataclasses import dataclass


@dataclass(slots=True)
class SFTConfig:
    pretrained_checkpoint: str
    dataset_path: str
    checkpoint_directory: str

    batch_size: int = 8

    learning_rate: float = 1e-5
    weight_decay: float = 0.01

    beta1: float = 0.9
    beta2: float = 0.95

    gradient_clip: float = 1.0

    max_steps: int = 300
    log_every: int = 10
    checkpoint_every: int = 100

    sequence_length: int = 128
