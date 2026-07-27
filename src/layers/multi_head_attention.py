from typing import cast

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        max_sequence_length: int,
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

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

        self.projection = nn.Linear(
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
    ) -> torch.Tensor:
        B, T, D = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        print("After Linear:")
        print("Q:", q.shape)
        print("K:", k.shape)
        print("V:", v.shape)

        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        print("\nAfter View:")
        print("Q:", q.shape)
        print("K:", k.shape)
        print("V:", v.shape)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        print("\nAfter Transpose:")
        print("Q:", q.shape)
        print("K:", k.shape)
        print("V:", v.shape)

        scale = self.head_dim**-0.5
        scores = (q @ k.transpose(-2, -1)) * scale

        print("\nScores:")
        print(scores.shape)

        mask = cast(torch.Tensor, self.mask)[:T, :T]

        scores = scores.masked_fill(
            mask == 0,
            float("-inf"),
        )

        weights = torch.softmax(scores, dim=-1)

        print("\nWeights:")
        print(weights.shape)

        output = weights @ v

        print("\nOutput after attention:")
        print(output.shape)

        output = output.transpose(1, 2)

        print("\nAfter transpose back:")
        print(output.shape)

        output = output.contiguous().view(
            B,
            T,
            self.embedding_dim,
        )

        output = self.projection(output)

        return output
