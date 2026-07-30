import torch
from torch import nn

from src.configs.gpt_config import GPTConfig
from src.layers.feed_forward import FeedForward
from src.layers.multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        config: GPTConfig,
    ) -> None:
        super().__init__()

        # self.embedding_dim: int = embedding_dim
        # self.num_heads: int = num_heads
        # self.max_sequence_length: int = max_sequence_length

        self.attention_norm = nn.LayerNorm(normalized_shape=config.embedding_dim)

        self.multi_head_attention = MultiHeadAttention(config=config)

        self.feed_forward_norm = nn.LayerNorm(normalized_shape=config.embedding_dim)

        self.feed_forward = FeedForward(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # normalized_x = self.attention_norm(x)

        # attention_output = self.multi_head_attention(normalized_x)

        # x = x + attention_output

        # normalized_x = self.feed_forward_norm(x)

        # feed_forward_output = self.feed_forward(normalized_x)

        # x = x + feed_forward_output

        # return x

        residual = hidden_states

        hidden_states = self.attention_norm(hidden_states)
        hidden_states = self.multi_head_attention(hidden_states)

        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.feed_forward_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states
