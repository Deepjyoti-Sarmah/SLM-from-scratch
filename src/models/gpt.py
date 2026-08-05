import torch
import torch.nn.functional as F
from torch import nn

from src.configs.gpt_config import GPTConfig
from src.embeddings.embedding_layer import EmbeddingLayer
from src.layers.decoder_block import DecoderBlock


class GPT(nn.Module):
    def __init__(
        self,
        *,
        config: GPTConfig,
    ) -> None:
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size

        self.input_embedding = EmbeddingLayer(config=config)

        self.blocks = nn.ModuleList(
            [DecoderBlock(config=config) for _ in range(config.num_layers)]
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

    @torch.no_grad()
    def generate(
        self,
        *,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if temperature < 0.0:
            raise ValueError("temperature must be greater than or equal to 0")

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        self.eval()

        for _ in range(max_new_tokens):
            input_ids = token_ids[
                :,
                -self.config.max_sequence_length :,
            ]

            logits, _ = self(
                token_ids=input_ids,
            )

            # next_token_logits = logits[:, -1, :]

            # next_token = torch.argmax(
            #     next_token_logits,
            #     dim=-1,
            # )

            next_token_logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token = torch.argmax(
                    next_token_logits,
                    dim=-1,
                )
            else:
                next_token_logits = next_token_logits / temperature

                if top_k is not None:
                    k = min(
                        top_k,
                        self.vocab_size,
                    )

                    top_k_value, _ = torch.topk(
                        next_token_logits,
                        k=k,
                    )

                    threashold = top_k_value[:, -1].unsqueeze(-1)

                    next_token_logits = next_token_logits.masked_fill(
                        next_token_logits < threashold,
                        float("-inf"),
                    )

                probabilities = torch.softmax(
                    next_token_logits,
                    dim=-1,
                )

                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1,
                ).squeeze(-1)

            token_ids = torch.cat(
                (
                    token_ids,
                    next_token.unsqueeze(1),
                ),
                dim=1,
            )

        return token_ids
