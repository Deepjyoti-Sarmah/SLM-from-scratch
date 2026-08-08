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

    training_config = TrainingConfig()

    pipeline = build_training_pipeline(
        tokenizer=tokenizer,
        token_ids=token_ids,
        model_config=model_config,
        training_config=training_config,
    )

    pipeline.trainer.train()

    print(
        pipeline.generator.generate(
            prompt="ROMEO",
            max_new_tokens=300,
        )
    )


if __name__ == "__main__":
    main()
