import torch
import torch.nn.functional as F
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

        self.config = config
        self.vocab_size = config.vocab_size

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

        self.apply(self._init_weights)

        self._tie_weights()

    def _init_weights(
        self,
        module: nn.Module,
    ) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )

    def _tie_weights(self) -> None:
        self.lm_head.weight = self.input_embedding.token_embedding.embedding.weight

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden_states = self.input_embedding(token_ids)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.final_norm(hidden_states)

        logits = self.lm_head(hidden_states)

        loss: torch.Tensor | None = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )

        return logits, loss
