import torch

from src.configs.training_config import TrainingConfig
from src.models.gpt import GPT


def build_optimizer(
    *,
    model: GPT,
    config: TrainingConfig,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        params=model.parameters(),
        lr=config.learning_rate,
        betas=(
            config.beta1,
            config.beta2,
        ),
        weight_decay=config.weight_decay,
    )
