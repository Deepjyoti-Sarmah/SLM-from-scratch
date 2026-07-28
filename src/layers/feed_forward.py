import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
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

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        hidden = self.input_projection(x)

        print("After Input projection:")
        print(hidden.shape)

        hidden = self.activation(hidden)

        print("\nAfter GELU:")
        print(hidden.shape)

        output = self.output_projection(hidden)

        print("\nAfter Output Projection:")
        print(output.shape)

        return output
