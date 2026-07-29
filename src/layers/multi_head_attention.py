import torch
from torch import nn

_MASK_FILL_VALUE = float("-inf")


class MultiHeadAttention(nn.Module):
    mask: torch.Tensor

    def __init__(
        self,
        *,
        embedding_dim: int,
        num_heads: int,
        max_sequence_length: int,
    ) -> None:
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim: int = embedding_dim
        self.num_heads: int = num_heads
        self.head_dim: int = embedding_dim // num_heads

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

        self.output_projection = nn.Linear(
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

    def _split_heads(
        self,
        tensor: torch.Tensor,
        *,
        batch_size: int,
        sequence_length: int,
    ) -> torch.Tensor:
        """
        Convert
            (B, T, D)
        into
            (B, H, T, Dh)
        """

        return tensor.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape

        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        print("After Linear:")
        print("Q:", query.shape)
        print("K:", key.shape)
        print("V:", value.shape)

        query = self._split_heads(
            query,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        key = self._split_heads(
            key,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        value = self._split_heads(
            value,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )

        print("\nAfter Split Heads:")
        print("Q:", query.shape)
        print("K:", key.shape)
        print("V:", value.shape)

        scale = self.head_dim**-0.5
        attention_scores = (query @ key.transpose(-2, -1)) * scale

        print("\nAttention Scores:")
        print(attention_scores.shape)

        causal_mask = self.mask[:sequence_length, :sequence_length]

        attention_scores = attention_scores.masked_fill(
            causal_mask == 0,
            _MASK_FILL_VALUE,
        )

        attention_weights = torch.softmax(attention_scores, dim=-1)

        print("\nAttention Weights:")
        print(attention_scores.shape)

        attention_output = attention_weights @ value

        print("\nAttention Output:")
        print(attention_output.shape)

        attention_output = attention_output.transpose(1, 2)

        print("\nAfter Merge Transpose:")
        print(attention_output.shape)

        attention_output = attention_output.contiguous().view(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )

        print("\nAfter Merge Heads:")
        print(attention_output.shape)

        attention_output = self.output_projection(attention_output)

        print("\nAfter Output Projection:")
        print(attention_output.shape)

        return attention_output
