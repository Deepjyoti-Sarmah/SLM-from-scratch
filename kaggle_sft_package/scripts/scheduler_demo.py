import torch
from torch import nn

from src.configs.training_config import TrainingConfig
from src.training.scheduler import build_scheduler


def main() -> None:
    model = nn.Linear(
        in_features=10,
        out_features=10,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    config = TrainingConfig(
        learning_rate=3e-4,
        warmup_steps=2,
        max_steps=20,
        minimum_learning_rate=3e-5,
    )

    scheduler = build_scheduler(
        optimizer=optimizer,
        config=config,
    )

    for step in range(1, 21):
        optimizer.step()
        scheduler.step()

        learning_rate = optimizer.param_groups[0]["lr"]

        print(f"Step {step:02d} | LR: {learning_rate:.8f}")


if __name__ == "__main__":
    main()
