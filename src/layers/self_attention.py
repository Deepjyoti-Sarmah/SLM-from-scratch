from typing import cast

import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        max_sequence_length: int,
    ):
        super().__init__()

        self.query = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            bias=False,
        )
        self.key = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            bias=False,
        )
        self.value = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            bias=False,
        )

        mask = torch.tril(
            torch.ones(
                max_sequence_length,
                max_sequence_length,
            )
        )

        self.register_buffer(
            "mask",
            mask,
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # scores = query @ key.transpose(-2, -1)
        # scores = scores / math.sqrt(query.size(-1))

        # scores = (query @ key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        scale = query.size(-1) ** -0.5
        scores = (query @ key.transpose(-2, -1)) * scale

        T = x.size(1)

        # mask = torch.tril(torch.ones(T, T, device=x.device))
        mask = cast(torch.Tensor, self.mask)[:T, :T]

        scores = scores.masked_fill(
            mask == 0,
            float("-inf"),
        )

        weights = torch.softmax(scores, dim=-1)

        # TODO: temp
        print(weights[0])

        output = weights @ value

        return output
