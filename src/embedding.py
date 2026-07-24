import torch
from torch import nn


class MyEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)
