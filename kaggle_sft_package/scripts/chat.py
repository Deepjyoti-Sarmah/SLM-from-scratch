"""Interactive terminal chat with the from-scratch Shakespeare GPT.

Usage:

    uv run python scripts/chat.py

Options can be viewed with:

    uv run python scripts/chat.py --help
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.configs.gpt_config import GPTConfig
from src.inference.generator import TextGenerator
from src.models.gpt import GPT
from src.tokenization.char_tokenizer import CharacterTokenizer

CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "step_010000.pt"
DATA_PATH = REPO_ROOT / "data" / "tiny_shakespeare.txt"

MODEL_CONFIG: dict[str, int | float] = {
    "vocab_size": 65,
    "max_sequence_length": 128,
    "embedding_dim": 256,
    "num_heads": 8,
    "num_layers": 6,
    "dropout_probability": 0.0,
}

PROMPT_CONTEXT = 112

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_MAGENTA = "\033[35m"
_RED = "\033[31m"

_TTY = sys.stdout.isatty()

_COLOR = _TTY and not os.environ.get("NO_COLOR")


def _paint(
    text: str,
    code: str,
) -> str:
    if not _COLOR:
        return text
    return f"{code}{text}{_RESET}"


def _load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[GPT, CharacterTokenizer, TextGenerator]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    config = GPTConfig(**MODEL_CONFIG)

    model = GPT(config=config)

    result = model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )

    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"state dict mismatch: missing={result.missing_keys} unexpected={result.unexpected_keys}"
        )

    model.to(device)
    model.eval()

    text = DATA_PATH.read_text(encoding="utf-8")

    tokenizer = CharacterTokenizer(text)

    generator = TextGenerator(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    return model, tokenizer, generator


def _select_device(
    device_flag: str | None,
) -> tuple[torch.device, str]:
    if device_flag is not None:
        if device_flag.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this machine.")
        return torch.device(device_flag), device_flag

    if torch.cuda.is_available():
        return torch.device("cuda"), f"cuda ({torch.cuda.get_device_name(0)})"

    return torch.device("cpu"), "cpu"


def _random_prompt() -> str:
    lines = DATA_PATH.read_text(encoding="utf-8").splitlines()

    candidates = [
        line.strip()
        for line in lines
        if line.strip()
        and line[0].isupper()
        and ":" in line
        and len(line.strip()) < 60
    ]

    return random.choice(candidates) if candidates else "ROMEO:"


def _print_banner(
    model: GPT,
    device_label: str,
) -> None:
    parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    print(_paint("=" * 72, _BOLD))
    print(_paint("  SLM from scratch -- terminal GPT chat", _BOLD))
    print(_paint("=" * 72, _BOLD))
    print(f"  Parameters:    {parameters:,}")
    print(f"  Tokenizer:     character-level (vocab {model.config.vocab_size})")
    print(
        f"  Architecture:  {model.config.num_layers} layers | {model.config.num_heads} heads | "
        f"{model.config.embedding_dim} embed | {model.config.max_sequence_length} context"
    )
    print(f"  Checkpoint:    {CHECKPOINT_PATH.name} (10,000 steps)")
    print(f"  Device:        {device_label}")
    print(_paint("-" * 72, _DIM))
    print(
        _paint(
            "  This is a text-continuation model trained on Tiny Shakespeare.\n"
            "  It does not answer questions -- it continues your text in a\n"
            "  Shakespearean style. Type anything and watch it write.",
            _DIM,
        )
    )
    print(_paint("  Type /help for commands.  Type /quit to exit.", _DIM))
    print()


def _append_turn(
    transcript: str,
    user_message: str,
) -> str:
    return f"{transcript}\nYou: {user_message}\nGPT: "


def _build_prompt(
    transcript: str,
    context: int = PROMPT_CONTEXT,
) -> str:
    if len(transcript) <= context:
        return transcript

    tail = transcript[-context:]

    marker = tail.rfind("\nYou: ")

    if marker != -1:
        return tail[marker + 1 :]

    return tail


def _generate_reply(
    generator: TextGenerator,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    seed: int | None,
) -> str:
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    full_text = generator.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    reply = full_text[len(prompt) :]

    return reply


def _stream_print(
    text: str,
    *,
    delay: float,
) -> None:
    sys.stdout.write(_paint("GPT> ", _GREEN))

    for character in text:
        sys.stdout.write(_paint(character, _GREEN))
        sys.stdout.flush()
        if delay:
            time.sleep(delay)

    sys.stdout.write("\n")
    sys.stdout.flush()


def _print_help() -> None:
    print(_paint("Commands", _BOLD))
    print("  /help              show this help")
    print("  /reset             clear the conversation")
    print("  /temp <value>      set temperature (0.0 = greedy)")
    print("  /top_k <value>     set top-k filter (none to disable)")
    print("  /top_p <value>     set top-p filter (none to disable)")
    print("  /new <tokens>      set tokens generated per reply")
    print("  /params            show current sampling parameters")
    print("  /model             show model info")
    print("  /quit, /exit       leave the chat")
    print()
    print(_paint("Sampling defaults", _BOLD))
    print("  temperature 0.8 | top_k 40 | top_p 0.9 | 160 new tokens")
    print()
    print(_paint("Tips", _BOLD))
    print("  Enter an empty line to regenerate the last reply.")
    print("  Ctrl+C interrupts generation, Ctrl+D exits.")


def _print_params(
    *,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    max_new_tokens: int,
) -> None:
    print(f"  temperature = {temperature}")
    print(f"  top_k       = {top_k}")
    print(f"  top_p       = {top_p}")
    print(f"  new tokens  = {max_new_tokens}")


def run_chat(
    generator: TextGenerator,
    args: argparse.Namespace,
) -> None:
    transcript = ""
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    max_new_tokens = args.tokens
    delay = 0.012 if not args.no_stream else 0.0

    while True:
        try:
            raw = input(_paint("You> ", _CYAN))
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            print(_paint("(interrupted)", _DIM))
            continue

        message = raw.strip()

        if not message:
            if not transcript:
                continue
            prompt = _build_prompt(transcript)
            print(_paint("(regenerating previous reply)", _DIM))
        elif message.startswith("/"):
            command, _, argument = message.partition(" ")

            if command in ("/quit", "/exit"):
                print(_paint("Goodbye.", _DIM))
                break

            if command == "/help":
                _print_help()
                continue

            if command == "/reset":
                transcript = ""
                print(_paint("Conversation cleared.", _DIM))
                continue

            if command in ("/temp", "/top_k", "/top_p", "/new"):
                try:
                    if argument.strip().lower() in ("none", ""):
                        value = None
                    else:
                        value = float(argument) if command in ("/temp", "/top_p") else int(argument)
                except ValueError:
                    print(_paint(f"Could not parse number: {argument!r}", _RED))
                    continue

                if command == "/temp":
                    temperature = value if value is not None else 0.8
                elif command == "/top_k":
                    top_k = value
                elif command == "/top_p":
                    top_p = value
                else:
                    max_new_tokens = value if value is not None else 160

                print(_paint(f"{command.strip('/')} -> {value}", _DIM))
                continue

            if command == "/params":
                _print_params(
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                )
                continue

            if command == "/model":
                _print_banner(generator.model, "chat")
                continue

            print(_paint(f"Unknown command: {command} (try /help)", _RED))
            continue
        else:
            transcript = _append_turn(
                transcript[-4096:],
                message,
            )

        prompt = _build_prompt(transcript)

        if _TTY:
            print(_paint("(typing...)", _DIM), end="\r")
            sys.stdout.flush()

        try:
            reply = _generate_reply(
                generator,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=args.seed,
            )
        except KeyboardInterrupt:
            print(_paint("(generation interrupted)", _DIM))
            continue

        reply = reply.strip()

        transcript += reply

        if _TTY:
            sys.stdout.write("\033[K")
            sys.stdout.flush()

        _stream_print(reply, delay=delay)


def run_oneshot(
    generator: TextGenerator,
    args: argparse.Namespace,
) -> None:
    prompt = args.prompt or _random_prompt()

    print(_paint("PROMPT> ", _YELLOW) + prompt)
    print()

    reply = _generate_reply(
        generator,
        prompt,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )

    delay = 0.012 if not args.no_stream else 0.0

    _stream_print(reply.strip(), delay=delay)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Talk to the from-scratch Shakespeare GPT in the terminal.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help="Path to the training checkpoint.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device: 'cpu', 'cuda', or 'cuda:N'.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0.0 = greedy).",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k sampling filter.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus (top-p) sampling filter.",
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=160,
        help="Number of characters generated per reply.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible sampling.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Run a single generation with this prompt, then exit.",
    )

    parser.add_argument(
        "--random-prompt",
        action="store_true",
        help="Run a single generation with a random prompt, then exit.",
    )

    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Print each reply at once instead of character-by-character.",
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in the output.",
    )

    return parser


def main() -> None:
    parser = build_parser()

    args = parser.parse_args()

    if args.no_color:
        global _COLOR
        _COLOR = False

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device, device_label = _select_device(args.device)

    if device.type == "cuda":
        print(_paint(f"Loading checkpoint on {device_label}...", _DIM))
    else:
        print(_paint("Loading checkpoint on CPU...", _DIM))

    model, tokenizer, generator = _load_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    _print_banner(model, device_label)

    if args.prompt is not None or args.random_prompt:
        run_oneshot(generator, args)
        return

    run_chat(generator, args)


if __name__ == "__main__":
    main()