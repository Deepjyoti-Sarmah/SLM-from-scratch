"""Download the Shakespeare GPT from the Hugging Face Hub and run inference.

    python download_and_run.py --model-id Deepjyoti/shakespeare-GPT
    python download_and_run.py \
        --model-id Deepjyoti/shakespeare-GPT \
        --prompt "To be or not to be" \
        --max-new-tokens 300 \
        --temperature 0.8
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "model.py",
    "tokenizer.py",
    "inference.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and run the Shakespeare GPT from the Hugging Face Hub.",
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default="Deepjyoti/shakespeare-GPT",
        help="Hugging Face repo id of the model.",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="shakespeare-gpt-cache",
        help="Directory where downloaded files are stored.",
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

    return parser.parse_args()


def _download(
    *,
    model_id: str,
    cache_dir: str,
) -> Path:
    cache = Path(cache_dir)

    cache.mkdir(
        parents=True,
        exist_ok=True,
    )

    for filename in REPO_FILES:
        hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=cache,
        )

    return cache


def main() -> None:
    args = _parse_args()

    cache = _download(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
    )

    inference_path = cache / "inference.py"

    spec = importlib.util.spec_from_file_location("deploy_inference", inference_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {inference_path}")

    inference = importlib.util.module_from_spec(spec)

    sys.argv = [
        str(inference_path),
        f"--local-dir={cache}",
        f"--prompt={args.prompt}",
        f"--max-new-tokens={args.max_new_tokens}",
        f"--temperature={args.temperature}",
        f"--device={args.device}",
    ]

    if args.top_k is not None:
        sys.argv.append(f"--top-k={args.top_k}")

    if args.top_p is not None:
        sys.argv.append(f"--top-p={args.top_p}")

    if args.seed is not None:
        sys.argv.append(f"--seed={args.seed}")

    spec.loader.exec_module(inference)

    inference.main()


if __name__ == "__main__":
    main()