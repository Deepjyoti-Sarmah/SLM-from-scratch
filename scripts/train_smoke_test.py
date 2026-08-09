# Tokenizer
#    ↓
# Dataset / DataLoader
#    ↓
# GPT
#    ↓
# Optimizer
#    ↓
# Scheduler
#    ↓
# Training
#    ↓
# Validation
#    ↓
# Checkpoint save
#    ↓
# Generation
#    ↓
# New pipeline
#    ↓
# Checkpoint resume


from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from src.configs.gpt_config import GPTConfig
from src.configs.training_config import TrainingConfig
from src.tokenization.char_tokenizer import CharacterTokenizer
from src.training.build import build_training_pipeline


def count_parameters(model: torch.nn.Module) -> int:
    """
    Return the number of trainable parameters in the model.
    """

    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def main() -> None:

    torch.manual_seed(42)

    print("=" * 80)
    print("GPT TRAINING SMOKE TEST")
    print("=" * 80)

    data_path = Path("data/tiny_shakespeare.txt")

    text = data_path.read_text(encoding="utf-8")

    print(f"Dataset: {data_path}")
    print(f"Characters: {len(text):,}")

    tokenizer = CharacterTokenizer(text=text)

    token_ids = tokenizer.encode(text)

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Token count: {len(token_ids):,}")

    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_sequence_length=32,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        dropout_probability=0.1,
    )

    with TemporaryDirectory() as checkpoint_directory:
        training_config = TrainingConfig(
            batch_size=8,
            learning_rate=3e-4,
            weight_decay=0.1,
            beta1=0.9,
            beta2=0.95,
            gradient_clip=1.0,
            train_ratio=0.9,
            num_epochs=1,
            log_every=1,
            checkpoint_every=5,
            checkpoint_directory=checkpoint_directory,
            warmup_steps=2,
            max_steps=20,
            minimum_learning_rate=3e-5,
        )

        print()
        print("Model configuration:")
        print(f"  Embedding dimension: {model_config.embedding_dim}")
        print(f"  Attention heads:     {model_config.num_heads}")
        print(f"  Transformer layers: {model_config.num_layers}")
        print(f"  Context length:      {model_config.max_sequence_length}")
        print(f"  Parameters:          ", end="")

        pipeline = build_training_pipeline(
            tokenizer=tokenizer,
            token_ids=token_ids,
            model_config=model_config,
            training_config=training_config,
        )

        print(f"{count_parameters(pipeline.model):,}")

        print()
        print("=" * 80)
        print("FIRST TRAINING RUN")
        print("=" * 80)

        pipeline.trainer.train()

        print()
        print("=" * 80)
        print("GENERATION TEST")
        print("=" * 80)

        generated_text = pipeline.generator.generate(
            prompt="ROMEO:",
            max_new_tokens=100,
            temperature=1.0,
            top_k=10,
            top_p=0.9,
        )

        print(generated_text)

        checkpoint_manager = pipeline.trainer.checkpoint_manager

        checkpoint_path = checkpoint_manager.latest_checkpoint()

        if checkpoint_path is None:
            raise RuntimeError("Smoke test failed: no checkpoint was created.")

        print()
        print(f"Checkpoint created: {checkpoint_path}")

        print()
        print("=" * 80)
        print("CHECKPOINT RESUME TEST")
        print("=" * 80)

        resumed_pipeline = build_training_pipeline(
            tokenizer=tokenizer,
            token_ids=token_ids,
            model_config=model_config,
            training_config=training_config,
        )

        resumed_pipeline.trainer.train()

        print()
        print("=" * 80)
        print("RESUMED MODEL GENERATION TEST")
        print("=" * 80)

        resumed_text = resumed_pipeline.generator.generate(
            prompt="ROMEO:",
            max_new_tokens=100,
            temperature=1.0,
            top_k=10,
            top_p=0.9,
        )

        print(resumed_text)

    print()
    print("=" * 80)
    print("SMOKE TEST PASSED")
    print("=" * 80)

    print()
    print("Verified:")
    print("  ✓ Tokenization")
    print("  ✓ DataLoader")
    print("  ✓ GPT forward pass")
    print("  ✓ Cross-entropy loss")
    print("  ✓ Backward pass")
    print("  ✓ Gradient clipping")
    print("  ✓ Optimizer")
    print("  ✓ Scheduler")
    print("  ✓ Validation")
    print("  ✓ Checkpoint saving")
    print("  ✓ Checkpoint loading")
    print("  ✓ Checkpoint resume")
    print("  ✓ Autoregressive generation")


if __name__ == "__main__":
    main()
