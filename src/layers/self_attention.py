# from typing import cast

# import torch
# from torch import nn


# class SelfAttention(nn.Module):
#     def __init__(
#         self,
#         embedding_dim: int,
#         max_sequence_length: int,
#     ):
#         super().__init__()

#         self.query = nn.Linear(
#             in_features=embedding_dim,
#             out_features=embedding_dim,
#             bias=False,
#         )
#         self.key = nn.Linear(
#             in_features=embedding_dim,
#             out_features=embedding_dim,
#             bias=False,
#         )
#         self.value = nn.Linear(
#             in_features=embedding_dim,
#             out_features=embedding_dim,
#             bias=False,
#         )

#         mask = torch.tril(
#             torch.ones(
#                 max_sequence_length,
#                 max_sequence_length,
#             )
#         )

#         self.register_buffer(
#             "mask",
#             mask,
#         )

#     def forward(
#         self,
#         x: torch.Tensor,
#     ):
#         query = self.query(x)
#         key = self.key(x)
#         value = self.value(x)

#         # scores = query @ key.transpose(-2, -1)
#         # scores = scores / math.sqrt(query.size(-1))

#         # scores = (query @ key.transpose(-2, -1)) / math.sqrt(query.size(-1))

#         scale = query.size(-1) ** -0.5
#         scores = (query @ key.transpose(-2, -1)) * scale

#         T = x.size(1)

#         # mask = torch.tril(torch.ones(T, T, device=x.device))
#         mask = cast(torch.Tensor, self.mask)[:T, :T]

#         scores = scores.masked_fill(
#             mask == 0,
#             float("-inf"),
#         )

#         weights = torch.softmax(scores, dim=-1)

#         # TODO: temp
#         print(weights[0])

#         output = weights @ value

#         return output

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
