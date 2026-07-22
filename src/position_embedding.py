from torch import nn


class PositionEmbedding(nn.Module):
    def __init__(
        self,
        max_sequence_length: int,
        embedding_dim: int,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=max_sequence_length,
            embedding_dim=embedding_dim,
        )

    def forward(self, max_sequence_length: int):
        return self.embedding(max_sequence_length)
