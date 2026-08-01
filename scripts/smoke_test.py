import torch

from src.configs.gpt_config import GPTConfig
from src.models.gpt import GPTModel


def main() -> None:
    config = GPTConfig(
        vocab_size=100,
        max_sequence_length=16,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
    )

    model = GPTModel(
        config=config,
    )

    token_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, config.max_sequence_length),
    )

    targets = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, config.max_sequence_length),
    )

    logits, loss = model(
        token_ids=token_ids,
        targets=targets,
    )

    print(f"Logits Shape : {logits.shape}")
    print(f"Loss         : {loss.item():.4f}")

    loss.backward()

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    print(f"Parameters   : {trainable_parameters:,}")

    print("✅ Smoke test passed!")

    print(f"Logits Shape : {logits.shape}")
    print(f"Loss         : {loss.item():.4f}")

    print(f"Min Logit    : {logits.min().item():.4f}")
    print(f"Max Logit    : {logits.max().item():.4f}")
    print(f"Mean Logit   : {logits.mean().item():.4f}")
    print(f"Std Logit    : {logits.std().item():.4f}")

    embedding = model.input_embedding.token_embedding.embedding.weight

    print(f"Embedding std : {embedding.std().item():.4f}")

    print(f"LM Head std   : {model.lm_head.weight.std().item():.4f}")

    print(f"Config initializer_range: {config.initializer_range}")
    print(f"Model initializer_range : {model.config.initializer_range}")


if __name__ == "__main__":
    main()