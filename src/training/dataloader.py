from torch.utils.data import DataLoader

from src.datasets.gpt_dataset import GPTDataset


def build_dataloader(
    *,
    token_ids: list[int],
    sequence_length: int,
    batch_size: int,
) -> DataLoader:

    dataset = GPTDataset(
        token_ids=token_ids,
        sequence_length=sequence_length,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )
