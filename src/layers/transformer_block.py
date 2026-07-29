import torch
from torch import nn

from main import feed_forward
from src.layers.feed_forward import FeedForward
from src.layers.layer_norm import LayerNorm
from src.layers.multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        num_heads: int,
        max_sequence_length: int,
    ) -> None:
        super().__init__()

        self.embedding_dim: int = embedding_dim
        self.num_heads: int = num_heads
        self.max_sequence_length: int = max_sequence_length

        self.attention_layer_norm = LayerNorm(embedding_dim=embedding_dim)

        self.multi_head_attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
        )

        self.feed_forward_layer_norm = LayerNorm(embedding_dim=embedding_dim)

        self.feed_forward = FeedForward(embedding_dim=embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        normalized_x = self.attention_layer_norm(x)

        attention_output = self.multi_head_attention(normalized_x)

        x = x + attention_output

        normalized_x = self.feed_forward_layer_norm(x)

        feed_forward_output = self.feed_forward(normalized_x)

        x = x + feed_forward_output

        return x
