import torch
from torch import nn

_MASK_FILL_VALUE = float("-inf")


class SelfAttention(nn.Module):
    mask: torch.Tensor

    def __init__(
        self,
        *,
        embedding_dim: int,
        max_sequence_length: int,
    ) -> None:
        super().__init__()

        self.embedding_dim: int = embedding_dim

        self.query_projection = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            bias=False,
        )

        self.key_projection = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            bias=False,
        )

        self.value_projection = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            bias=False,
        )

        self.register_buffer(
            "mask",
            self._create_causal_mask(
                max_sequence_length=max_sequence_length,
            ),
        )

    @staticmethod
    def _create_causal_mask(
        *,
        max_sequence_length: int,
    ) -> torch.Tensor:
        return torch.tril(
            torch.ones(
                max_sequence_length,
                max_sequence_length,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape

        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        print("After Linear:")
        print("Query :", query.shape)
        print("Key   :", key.shape)
        print("Value :", value.shape)

        scale = self.embedding_dim**-0.5

        attention_scores = (query @ key.transpose(-2, -1)) * scale

        print("\nAttention Scores:")
        print(attention_scores.shape)

        causal_mask = self.mask[
            :sequence_length,
            :sequence_length,
        ]

        attention_scores = attention_scores.masked_fill(
            causal_mask == 0,
            _MASK_FILL_VALUE,
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        print("\nAttention Weights:")
        print(attention_weights.shape)

        # Optional: inspect one attention matrix
        print(attention_weights[0])

        attention_output = attention_weights @ value

        print("\nAttention Output:")
        print(attention_output.shape)

        return attention_output
