"""Convert the training checkpoint into a deployment-only safetensors file.

Loads the checkpoint ``model_state`` into the standalone deployment model,
verifies a strict load, then writes:

    deploy/shakespeare-gpt/model.safetensors
    deploy/shakespeare-gpt/config.json
    deploy/shakespeare-gpt/metadata.json

Training state (optimizer, scheduler, epoch, step) is intentionally
dropped. Shared tensors (tied lm_head / token embedding weights) are
preserved by safetensors.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEPLOYMENT_DIRECTORY = REPOSITORY_ROOT / "deploy" / "shakespeare-gpt"
CHECKPOINT_PATH = REPOSITORY_ROOT / "checkpoints" / "step_010000.pt"

sys.path.insert(0, str(DEPLOYMENT_DIRECTORY))

from model import GPT, GPTConfig  # noqa: E402


def main() -> None:
    DEPLOYMENT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    config = GPTConfig(
        vocab_size=65,
        max_sequence_length=128,
        embedding_dim=256,
        num_heads=8,
        num_layers=6,
        dropout_probability=0.0,
    )

    model = GPT(config=config)

    result = model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )

    assert not result.missing_keys, result.missing_keys
    assert not result.unexpected_keys, result.unexpected_keys

    # Clone every tensor so the tied lm_head / token-embedding pair is
    # stored as independent copies. This keeps all 65 keys present in the
    # file (safetensors omits shared tensors), which lets a downstream
    # `load_file` + `load_state_dict(..., strict=True)` succeed.
    export_state = {
        name: tensor.detach().clone().contiguous()
        for name, tensor in model.state_dict().items()
    }

    weights_path = DEPLOYMENT_DIRECTORY / "model.safetensors"

    save_file(
        export_state,
        weights_path,
    )

    config_path = DEPLOYMENT_DIRECTORY / "config.json"

    config_path.write_text(
        json.dumps(
            config.to_json(),
            indent=2,
        ),
        encoding="utf-8",
    )

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    metadata = {
        "training_steps": checkpoint["global_step"],
        "training_epoch": checkpoint["epoch"],
        "parameters": parameters,
    }

    (DEPLOYMENT_DIRECTORY / "metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print("SAFETENSORS EXPORT")
    print("=" * 80)
    print(f"Checkpoint:  {CHECKPOINT_PATH}")
    print(f"Strict load: {result}")
    print(f"Weights:     {weights_path}")
    print(f"Config:      {config_path}")
    print(f"Metadata:    {DEPLOYMENT_DIRECTORY / 'metadata.json'}")
    print(f"Parameters:  {parameters:,}")
    print("=" * 80)


if __name__ == "__main__":
    main()