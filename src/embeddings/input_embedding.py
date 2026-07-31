import torch
from torch import nn

from src.configs.gpt_config import GPTConfig
from src.embeddings.position_embedding import PositionEmbedding
from src.embeddings.token_embedding import TokenEmbedding


class InputEmbedding(nn.Module):
    def __init__(
        self,
        *,
        config: GPTConfig,
    ):
        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
        )

        self.position_embedding = PositionEmbedding(
            max_sequence_length=config.max_sequence_length,
            embedding_dim=config.embedding_dim,
        )

        self.dropout = nn.Dropout(config.dropout_probability)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, sequence_length = token_ids.shape

        position_ids = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        token_embeddings = self.token_embedding(token_ids)
        position_embeddings = self.position_embedding(position_ids)

        input_embeddings = token_embeddings + position_embeddings

        input_embeddings = self.dropout(input_embeddings)

        return input_embeddings
