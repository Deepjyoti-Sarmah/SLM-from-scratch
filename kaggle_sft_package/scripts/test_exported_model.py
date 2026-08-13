import json
from pathlib import Path

import torch

from src.configs.gpt_config import GPTConfig
from src.models.gpt import GPT

ARTIFACT_DIRECTORY = Path("artifacts/shakespeare-gpt")


def load_tokenizer() -> tuple[dict[str, int], dict[int, str]]:
    data = json.loads(
        (ARTIFACT_DIRECTORY / "tokenizer.json").read_text(
            encoding="utf-8",
        )
    )

    token_to_id = {
        token: int(token_id) for token, token_id in data["token_to_id"].items()
    }

    id_to_token = {
        int(token_id): token for token_id, token in data["id_to_token"].items()
    }

    return token_to_id, id_to_token


def encode(
    text: str,
    token_to_id: dict[str, int],
) -> list[int]:
    return [token_to_id[token] for token in text]


def decode(
    token_ids: list[int],
    id_to_token: dict[int, str],
) -> str:
    return "".join(id_to_token[token_id] for token_id in token_ids)


@torch.no_grad()
def generate(
    *,
    model: GPT,
    token_ids: list[int],
    max_new_tokens: int,
    context_length: int,
    device: torch.device,
) -> list[int]:

    model.eval()

    tokens = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    for _ in range(max_new_tokens):
        context = tokens[:, -context_length:]

        logits, _ = model(
            token_ids=context,
        )

        next_token_logits = logits[:, -1, :]

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1,
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        tokens = torch.cat(
            [tokens, next_token],
            dim=1,
        )

    return tokens[0].tolist()


def main() -> None:
    print("=" * 80)
    print("EXPORTED MODEL TEST")
    print("=" * 80)

    config_data = json.loads(
        (ARTIFACT_DIRECTORY / "config.json").read_text(
            encoding="utf-8",
        )
    )

    token_to_id, id_to_token = load_tokenizer()

    print(f"Vocabulary size: {len(token_to_id)}")

    config = GPTConfig(
        vocab_size=config_data["vocab_size"],
        max_sequence_length=config_data["max_sequence_length"],
        embedding_dim=config_data["embedding_dim"],
        num_heads=config_data["num_heads"],
        num_layers=config_data["num_layers"],
        dropout_probability=config_data["dropout_probability"],
    )

    model = GPT(config=config)

    state_dict = torch.load(
        ARTIFACT_DIRECTORY / "model.pt",
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    print(f"Device: {device}")
    print("Model weights loaded successfully.")

    prompt = "ROMEO:"

    token_ids = encode(
        prompt,
        token_to_id,
    )

    generated_ids = generate(
        model=model,
        token_ids=token_ids,
        max_new_tokens=300,
        context_length=config.max_sequence_length,
        device=device,
    )

    generated_text = decode(
        generated_ids,
        id_to_token,
    )

    print()
    print("=" * 80)
    print("GENERATED TEXT")
    print("=" * 80)
    print(generated_text)
    print("=" * 80)


if __name__ == "__main__":
    main()
