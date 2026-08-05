from src.configs.gpt_config import GPTConfig
from src.configs.training_config import TrainingConfig
from src.configs.training_pipeline import TrainingPipeline
from src.models.gpt import GPT
from src.tokenization.char_tokenizer import CharacterTokenizer
from src.training.dataloader import build_dataloader
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.split import split_token_ids
from src.training.trainer import Trainer


def build_training_pipeline(
    *,
    tokenizer: CharacterTokenizer,
    token_ids: list[int],
    model_config: GPTConfig,
    training_config: TrainingConfig,
) -> TrainingPipeline:
    """
    Build the complete training pipeline.

    Text
        ↓
    Tokenizer
        ↓
    Token IDs
        ↓
    DataLoader
        ↓
    GPT
        ↓
    Optimizer
        ↓
    Scheduler
        ↓
    Trainer
    """

    model = GPT(config=model_config)

    train_token_ids, validation_token_ids = split_token_ids(
        token_ids=token_ids,
        train_ratio=training_config.train_ratio,
    )

    train_dataloader = build_dataloader(
        token_ids=train_token_ids,
        sequence_length=model_config.max_sequence_length,
        batch_size=training_config.batch_size,
    )

    validation_dataloader = build_dataloader(
        token_ids=validation_token_ids,
        sequence_length=model_config.max_sequence_length,
        batch_size=training_config.batch_size,
    )

    optimizer = build_optimizer(
        model=model,
        config=training_config,
    )

    scheduler = build_scheduler(
        optimizer=optimizer,
        config=training_config,
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=training_config,
    )

    return TrainingPipeline(
        trainer=trainer,
        model=model,
        tokenizer=tokenizer,
    )
