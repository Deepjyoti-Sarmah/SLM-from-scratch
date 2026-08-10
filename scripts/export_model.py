import json
from pathlib import Path

import torch

CHECKPOINT_PATH = Path("checkpoints/step_010000.pt")
OUTPUT_DIRECTORY = Path("artifacts/shakespeare-gpt")


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    model_state = checkpoint["model_state"]

    model_path = OUTPUT_DIRECTORY / "model.pt"

    torch.save(
        model_state,
        model_path,
    )

    config = {
        "model_type": "gpt",
        "vocab_size": 65,
        "max_sequence_length": 128,
        "embedding_dim": 256,
        "num_heads": 8,
        "num_layers": 6,
        "dropout_probability": 0.0,
    }

    config_path = OUTPUT_DIRECTORY / "config.json"

    config_path.write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata = {
        "training_steps": checkpoint["global_step"],
        "training_epoch": checkpoint["epoch"],
        "parameters": sum(parameter.numel() for parameter in model_state.values()),
    }

    metadata_path = OUTPUT_DIRECTORY / "metadata.json"

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print("MODEL EXPORT")
    print("=" * 80)
    print(f"Model:      {model_path}")
    print(f"Config:     {config_path}")
    print(f"Metadata:   {metadata_path}")
    print(f"Parameters: {metadata['parameters']:,}")
    print(f"Steps:      {metadata['training_steps']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
