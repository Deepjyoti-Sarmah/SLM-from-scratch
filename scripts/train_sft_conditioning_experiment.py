#!/usr/bin/env python3

"""Train a small TOPIC conditioning SFT experiment.

This intentionally does not touch the main SFT dataset or checkpoints. It starts
from checkpoints/step_010000.pt and writes to
checkpoints/sft_conditioning_experiment/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.optim.adamw import AdamW

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.configs.gpt_config import GPTConfig
from src.configs.sft_config import SFTConfig
from src.datasets.instruction_dataset import InstructionDataset
from src.inference.generator import TextGenerator
from src.models.gpt import GPT
from src.tokenization.char_tokenizer import CharacterTokenizer

TOPICS = [
    "AdamW",
    "Python",
    "API",
    "database",
    "recursion",
]


def load_model(
    *,
    checkpoint_path: str,
    model_config: GPTConfig,
    device: torch.device,
) -> GPT:
    model = GPT(config=model_config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    return model


def save_checkpoint(
    *,
    model: GPT,
    optimizer: AdamW,
    step: int,
    directory: str,
) -> Path:
    checkpoint_directory = Path(directory)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_directory / f"step_{step:06d}.pt"

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "global_step": step,
        },
        checkpoint_path,
    )

    print(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def evaluate_generation(
    *,
    model: GPT,
    tokenizer: CharacterTokenizer,
    device: torch.device,
    max_new_tokens: int = 90,
) -> None:
    generator = TextGenerator(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    model.eval()

    print()
    print("=" * 80)
    print("GREEDY TOPIC GENERATION")
    print("=" * 80)

    for topic in TOPICS:
        prompt = f"TOPIC: {topic}\nQ: What is it?\nA:"
        generated = generator.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=1,
        )

        answer = generated[len(prompt) :].strip()
        is_match = topic.lower() in answer.lower()

        print()
        print(f"PROMPT:\n{prompt}")
        print(f"OUTPUT:\n{generated}")
        print(f"CONCEPT_MATCH: {is_match}")

    model.train()


def main() -> int:
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = SFTConfig(
        pretrained_checkpoint="checkpoints/step_010000.pt",
        dataset_path="data/sft_conditioning_experiment.jsonl",
        checkpoint_directory="checkpoints/sft_conditioning_experiment",
        batch_size=5,
        learning_rate=1e-4,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.95,
        gradient_clip=1.0,
        max_steps=300,
        log_every=10,
        checkpoint_every=100,
        sequence_length=128,
    )

    print("=" * 80)
    print("SFT CONDITIONING EXPERIMENT")
    print("=" * 80)
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    tokenizer = CharacterTokenizer(
        Path("data/tiny_shakespeare.txt").read_text(encoding="utf-8")
    )
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_sequence_length=128,
        embedding_dim=256,
        num_heads=8,
        num_layers=6,
    )

    model = load_model(
        checkpoint_path=config.pretrained_checkpoint,
        model_config=model_config,
        device=device,
    )

    print(f"Base checkpoint loaded: {config.pretrained_checkpoint}")
    print(
        "Parameters:",
        f"{sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):,}",
    )

    dataset = InstructionDataset(
        path=config.dataset_path,
        tokenizer=tokenizer,
        sequence_length=config.sequence_length,
    )
    print(f"Experiment examples: {len(dataset)}")

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )

    model.train()
    global_step = 0
    latest_loss = None

    while global_step < config.max_steps:
        for input_ids, target_ids in dataloader:
            if global_step >= config.max_steps:
                break

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(token_ids=input_ids, targets=target_ids)
            assert loss is not None
            assert torch.isfinite(loss).item(), f"Non-finite loss: {loss.item()}"

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.gradient_clip,
            )
            optimizer.step()

            global_step += 1
            latest_loss = loss.item()

            if global_step % config.log_every == 0:
                print(f"Step {global_step:04d} | Loss: {latest_loss:.4f}")

            if global_step % config.checkpoint_every == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=global_step,
                    directory=config.checkpoint_directory,
                )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=global_step,
        directory=config.checkpoint_directory,
    )

    print()
    print(f"Final training loss: {latest_loss:.4f}" if latest_loss is not None else "No loss")

    evaluate_generation(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    print()
    print("Experiment complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
