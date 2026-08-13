import json
from pathlib import Path

from src.tokenization.char_tokenizer import CharacterTokenizer

DATA_PATH = Path("data/tiny_shakespeare.txt")
OUTPUT_PATH = Path("artifacts/shakespeare-gpt/tokenizer.json")


def main() -> None:
    text = DATA_PATH.read_text(
        encoding="utf-8",
    )

    tokenizer = CharacterTokenizer(text)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer_data = {
        "type": "character",
        "vocab_size": tokenizer.vocab_size,
        "token_to_id": tokenizer.token_to_id,
        "id_to_token": {
            str(token_id): token for token_id, token in tokenizer.id_to_token.items()
        },
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            tokenizer_data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print("TOKENIZER EXPORT")
    print("=" * 80)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Output:         {OUTPUT_PATH}")
    print()
    print("Vocabulary:")
    for token_id in range(tokenizer.vocab_size):
        token = tokenizer.id_to_token[token_id]
        print(f"{token_id:03d}: {token!r}")

    print("=" * 80)


if __name__ == "__main__":
    main()
