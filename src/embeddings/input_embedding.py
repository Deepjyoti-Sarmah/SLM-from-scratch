import torch
from torch import nn

from src.embeddings.position_embedding import PositionEmbedding
from src.embeddings.token_embedding import MyEmbedding


class InputEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        embedding_dim: int,
    ):
        super().__init__()

        self.token_embedding = MyEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.position_embedding = PositionEmbedding(
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape

        position_ids = torch.arange(
            T,
            device=token_ids.device,
        )

        token_embeddings = self.token_embedding(token_ids)
        position_embeddings = self.position_embedding(position_ids)

        input_embeddings = token_embeddings + position_embeddings

        return input_embeddings
