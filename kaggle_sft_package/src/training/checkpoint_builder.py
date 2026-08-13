from src.configs.training_config import TrainingConfig
from src.training.checkpoint import CheckpointManager


def build_checkpoint_manager(
    *,
    config: TrainingConfig,
) -> CheckpointManager:
    return CheckpointManager(
        checkpoint_directory=config.checkpoint_directory,
    )
