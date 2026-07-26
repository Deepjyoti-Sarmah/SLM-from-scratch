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

        return query, key, value
