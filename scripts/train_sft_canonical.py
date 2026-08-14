from __future__ import annotations

from pathlib import Path

import torch
from torch.optim.adamw import AdamW

from src.configs.gpt_config import GPTConfig
from src.configs.sft_config import SFTConfig
from src.datasets.instruction_dataset import InstructionDataset
from src.models.gpt import GPT
from src.tokenization.char_tokenizer import CharacterTokenizer


def load_base_model(
    *,
    checkpoint_path: str,
    model_config: GPTConfig,
    device: torch.device,
) -> GPT:

    model = GPT(
        config=model_config,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    result = model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )

    print("Base checkpoint loaded:")

    model.to(device)

    return model


def main() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("CANONICAL SFT EXPERIMENT")
    print("=" * 80)
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    config = SFTConfig(
        pretrained_checkpoint=("checkpoints/step_010000.pt"),
        dataset_path=("data/shakespeare_troll_sft_canonical.jsonl"),
        checkpoint_directory=("checkpoints/sft_canonical"),
        batch_size=8,
        learning_rate=1e-5,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.95,
        gradient_clip=1.0,
        max_steps=500,
        log_every=10,
        checkpoint_every=100,
        sequence_length=128,
    )

    data_path = Path("data/tiny_shakespeare.txt")

    text = data_path.read_text(encoding="utf-8")

    tokenizer = CharacterTokenizer(text)

    print(f"Vocabulary size: {tokenizer.vocab_size}")

    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_sequence_length=128,
        embedding_dim=256,
        num_heads=8,
        num_layers=6,
    )

    model = load_base_model(
        checkpoint_path=(config.pretrained_checkpoint),
        model_config=model_config,
        device=device,
    )

    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Parameters: {parameters:,}")

    dataset = InstructionDataset(
        path=config.dataset_path,
        tokenizer=tokenizer,
        sequence_length=config.sequence_length,
    )

    print(f"SFT examples: {len(dataset)}")

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(
            config.beta1,
            config.beta2,
        ),
        weight_decay=config.weight_decay,
    )

    model.train()

    global_step = 0

    while global_step < config.max_steps:
        for input_ids, target_ids in dataloader:
            if global_step >= config.max_steps:
                break

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad(set_to_none=True)

            _, loss = model(
                token_ids=input_ids,
                targets=target_ids,
            )

            assert loss is not None

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.gradient_clip,
            )

            optimizer.step()

            global_step += 1

            if global_step % config.log_every == 0:
                print(f"Step {global_step:04d} | Loss: {loss.item():.4f}")

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

    print("=" * 80)
    print(f"SFT training complete: {global_step} steps")
    print(f"Final checkpoint: {config.checkpoint_directory}/step_{global_step:06d}.pt")
    print("=" * 80)
    print(f"Maximum SFT steps reached: {config.max_steps}")


def save_checkpoint(
    *,
    model: GPT,
    optimizer: AdamW,
    step: int,
    directory: str,
) -> None:

    checkpoint_directory = Path(directory)

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

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


if __name__ == "__main__":
    main()