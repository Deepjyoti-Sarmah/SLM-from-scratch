"""Run inference with the Shakespeare GPT.

Usage (files already present next to this script):

    python inference.py --prompt "ROMEO:"

Usage (download weights/config/tokenizer from the Hub first):

    python inference.py --model-id Deepjyoti/shakespeare-GPT --prompt "ROMEO:"

Optional sampling flags (must match the model's generation interface):

    --max-new-tokens 200 --temperature 0.8 --top-k 40
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _resolve_repository(
    *,
    model_id: str | None,
    local_dir: str,
) -> Path:
    directory = Path(local_dir).resolve()

    if model_id is None:
        return directory

    from huggingface_hub import hf_hub_download

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for filename in (
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "model.py",
        "tokenizer.py",
    ):
        hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=directory,
        )

    return directory


def _load_module(
    name: str,
    path: Path,
):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec)

    sys.modules[name] = module

    spec.loader.exec_module(module)

    return module


def _select_device(
    device_flag: str | None,
) -> tuple[str, str | None]:
    if device_flag is not None:
        if device_flag.startswith("cuda") and not __import__("torch").cuda.is_available():
            raise RuntimeError("CUDA is not available on this machine.")
        return device_flag, None

    import torch

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return "cuda", gpu_name

    return "cpu", None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with the Shakespeare GPT.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="ROMEO:",
        help="Prompt text to condition generation on.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Number of new tokens to generate.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (0.0 = greedy).",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling filter.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus (top-p) sampling filter.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device: 'cpu', 'cuda', or a specific index such as 'cuda:0'.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible sampling.",
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Hugging Face repo id to download weights from.",
    )

    parser.add_argument(
        "--local-dir",
        type=str,
        default=".",
        help="Directory containing the model files (or download target).",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    directory = _resolve_repository(
        model_id=args.model_id,
        local_dir=args.local_dir,
    )

    tokenizer_module = _load_module(
        "tokenizer",
        directory / "tokenizer.py",
    )

    model_module = _load_module(
        "model",
        directory / "model.py",
    )

    config_data = json.loads(
        (directory / "config.json").read_text(
            encoding="utf-8",
        )
    )

    config = model_module.GPTConfig.from_json(config_data)

    tokenizer = tokenizer_module.CharacterTokenizer.from_file(
        directory / "tokenizer.json",
    )

    device, gpu_name = _select_device(args.device)

    import torch
    from safetensors.torch import load_file

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    model = model_module.GPT(config=config)

    state = load_file(
        directory / "model.safetensors",
        device=device,
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.to(device)
    model.eval()

    token_ids = tokenizer.encode(args.prompt)

    token_ids_tensor = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    generated = model.generate(
        token_ids=token_ids_tensor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    text = tokenizer.decode(generated[0].tolist())

    print(f"Device: {device}")

    if gpu_name is not None:
        print(f"GPU: {gpu_name}")

    print()
    print("PROMPT:")
    print(args.prompt)
    print()
    print("GENERATED:")
    print(text)


if __name__ == "__main__":
    main()