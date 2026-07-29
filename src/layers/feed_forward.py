import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        dropout_probability: float,
    ) -> None:
        super().__init__()

        hidden_dim = 4 * embedding_dim

        self.embedding_dim: int = embedding_dim
        self.hidden_dim: int = hidden_dim

        self.input_projection = nn.Linear(
            in_features=embedding_dim, out_features=hidden_dim
        )

        self.activation = nn.GELU()

        self.output_projection = nn.Linear(
            in_features=hidden_dim,
            out_features=embedding_dim,
        )

        self.dropout = nn.Dropout(dropout_probability)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        hidden_states = self.input_projection(hidden_states)

        hidden_states = self.activation(hidden_states)

        hidden_states = self.output_projection(hidden_states)

        hidden_states = self.dropout(hidden_states)

        return hidden_states
