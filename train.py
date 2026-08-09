from pathlib import Path

from src.configs.gpt_config import GPTConfig
from src.configs.training_config import TrainingConfig
from src.tokenization.char_tokenizer import CharacterTokenizer
from src.training.build import build_training_pipeline


def main() -> None:
    data_path = Path("data/tiny_shakespeare.txt")

    text = data_path.read_text(
        encoding="utf-8",
    )

    tokenizer = CharacterTokenizer(text)
    token_ids = tokenizer.encode(text)

    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_sequence_length=128,
        embedding_dim=256,
        num_heads=8,
        num_layers=6,
    )

    training_config = TrainingConfig(
        batch_size=64,
        learning_rate=3e-4,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.095,
        gradient_clip=1.0,
        train_ratio=0.9,
        num_epochs=100,
        log_every=100,
        checkpoint_every=1000,
        checkpoint_directory="checkpoints",
        warmup_steps=200,
        max_steps=10_000,
        minimum_learning_rate=3e-5,
    )

    pipeline = build_training_pipeline(
        tokenizer=tokenizer,
        token_ids=token_ids,
        model_config=model_config,
        training_config=training_config,
    )

    print("=" * 80)
    print("GPT TRAINING")
    print("=" * 80)
    print(f"Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"Token count:     {len(token_ids):,}")
    print(
        f"Parameters:      {sum(p.numel() for p in pipeline.model.parameters() if p.requires_grad):,}"
    )
    print(f"Batch size:      {training_config.batch_size}")
    print(f"Context length:  {model_config.max_sequence_length}")
    print(f"Max steps:       {training_config.max_steps:,}")
    print("=" * 80)

    pipeline.trainer.train()


if __name__ == "__main__":
    main()
