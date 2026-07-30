import torch
from torch import nn

from src.configs.gpt_config import GPTConfig
from src.embeddings.input_embedding import InputEmbedding
from src.layers.transformer_block import TransformerBlock


class GPTModel(nn.Module):
    def __init__(
        self,
        *,
        config: GPTConfig,
    ) -> None:
        super().__init__()

        self.input_embedding = InputEmbedding(config=config)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config=config) for _ in range(config.num_layers)]
        )

        self.final_norm = nn.LayerNorm(normalized_shape=config.embedding_dim)

        self.lm_head = nn.Linear(
            in_features=config.embedding_dim,
            out_features=config.vocab_size,
            bias=False,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.input_embedding(token_ids)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.final_norm(hidden_states)

        logits = self.lm_head(hidden_states)

        return logits
