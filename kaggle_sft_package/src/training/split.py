from __future__ import annotations


def split_token_ids(
    *,
    token_ids: list[int],
    train_ratio: float,
) -> tuple[list[int], list[int]]:

    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1.")

    split_index = int(len(token_ids) * train_ratio)

    train_token_ids = token_ids[:split_index]
    validation_token_ids = token_ids[split_index:]

    return train_token_ids, validation_token_ids
