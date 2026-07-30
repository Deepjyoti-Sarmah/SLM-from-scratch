import torch
from torch import nn

from src.configs.gpt_config import GPTConfig


class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        config: GPTConfig,
    ) -> None:
        super().__init__()

        hidden_dim = 4 * config.embedding_dim

        # self.embedding_dim: int = config.embedding_dim
        # self.hidden_dim: int = hidden_dim

        self.input_projection = nn.Linear(
            in_features=config.embedding_dim, out_features=hidden_dim
        )

        self.gelu = nn.GELU()

        self.output_projection = nn.Linear(
            in_features=hidden_dim,
            out_features=config.embedding_dim,
        )

        self.dropout = nn.Dropout(config.dropout_probability)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        hidden_states = self.input_projection(hidden_states)

        hidden_states = self.gelu(hidden_states)

        hidden_states = self.output_projection(hidden_states)

        hidden_states = self.dropout(hidden_states)

        return hidden_states
