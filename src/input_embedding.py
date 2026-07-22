import torch
from torch import nn


class InputEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        embedding_dim: int,
    ):
        super().__init__()
