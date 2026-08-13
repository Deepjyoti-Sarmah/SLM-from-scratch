import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.embedding_dim: int = embedding_dim
        self.eps: float = eps

        self.gamma = nn.Parameter(torch.zeros(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)

        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

        standard_deviation = torch.sqrt(variance + self.eps)

        normalized = (x - mean) / standard_deviation

        output = (self.gamma * normalized) + self.beta

        return output
