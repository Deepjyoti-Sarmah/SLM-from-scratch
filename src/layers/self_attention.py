import math

import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
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

        weights = torch.softmax(scores, dim=-1)

        output = weights @ value

        return output
