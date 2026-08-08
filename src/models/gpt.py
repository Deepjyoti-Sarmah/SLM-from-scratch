import torch
import torch.nn.functional as F
from torch import nn

from src.configs.gpt_config import GPTConfig
from src.embeddings.embedding_layer import EmbeddingLayer
from src.layers.decoder_block import DecoderBlock

_LOGIT_MASK_VALUE = float("-inf")


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
            flattened_logits = logits.reshape(
                -1,
                self.vocab_size,
            )

            flattened_targets = targets.reshape(-1)

            loss = F.cross_entropy(
                flattened_logits,
                flattened_targets,
            )

        return logits, loss

    def _apply_temperature(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
    ) -> torch.Tensor:
        if temperature == 0.0:
            return logits

        return logits / temperature

    def _apply_top_k(
        self,
        logits: torch.Tensor,
        *,
        top_k: int | None,
    ) -> torch.Tensor:
        if top_k is None:
            return logits

        k = min(
            top_k,
            self.vocab_size,
        )

        values, _ = torch.topk(
            logits,
            k=k,
        )

        threshold = values[:, -1].unsqueeze(-1)

        logits = logits.masked_fill(
            logits < threshold,
            _LOGIT_MASK_VALUE,
        )

        return logits

    def _apply_top_p(
        self,
        logits: torch.Tensor,
        *,
        top_p: float | None,
    ) -> torch.Tensor:
        if top_p is None:
            return logits

        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
            dim=-1,
        )

        sorted_probabilities = torch.softmax(
            sorted_logits,
            dim=-1,
        )

        cumulative_probilities = torch.cumsum(
            sorted_probabilities,
            dim=-1,
        )

        sorted_mask = cumulative_probilities > top_p

        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(
            sorted_mask,
            _LOGIT_MASK_VALUE,
        )

        filtered_logits = torch.full_like(
            logits,
            _LOGIT_MASK_VALUE,
        )

        filtered_logits.scatter_(
            dim=-1,
            index=sorted_indices,
            src=sorted_logits,
        )

        return filtered_logits

    def _prepare_logits(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
    ) -> torch.Tensor:
        """
        Apply all sampling strategies to the logits.
        """

        logits = self._apply_temperature(
            logits=logits,
            temperature=temperature,
        )

        logits = self._apply_top_k(
            logits=logits,
            top_k=top_k,
        )

        logits = self._apply_top_p(
            logits=logits,
            top_p=top_p,
        )

        return logits

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
    ) -> torch.Tensor:

        if temperature == 0.0:
            return torch.argmax(
                logits,
                dim=-1,
            )

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        return torch.multinomial(
            probabilities,
            num_samples=1,
        ).squeeze(-1)

    @torch.no_grad()
    def generate(
        self,
        *,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        # if temperature < 0.0:
        #     raise ValueError("temperature must be greater than or equal to 0")

        # if top_k is not None and top_k <= 0:
        #     raise ValueError("top_k must be greater than 0")

        was_training = self.training

        self.eval()

        if temperature < 0:
            raise ValueError("temperature must be greater than or equal to 0.")

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if top_p is not None and not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in the range (0, 1].")

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than 0.")

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

            # next_token_logits = logits[:, -1, :]

            # if temperature == 0.0:
            #     next_token = torch.argmax(
            #         next_token_logits,
            #         dim=-1,
            #     )
            # else:
            #     next_token_logits = next_token_logits / temperature

            #     if top_k is not None:
            #         k = min(
            #             top_k,
            #             self.vocab_size,
            #         )

            #         top_k_value, _ = torch.topk(
            #             next_token_logits,
            #             k=k,
            #         )

            #         threashold = top_k_value[:, -1].unsqueeze(-1)

            #         next_token_logits = next_token_logits.masked_fill(
            #             next_token_logits < threashold,
            #             float("-inf"),
            #         )

            #     probabilities = torch.softmax(
            #         next_token_logits,
            #         dim=-1,
            #     )

            #     next_token = torch.multinomial(
            #         probabilities,
            #         num_samples=1,
            #     ).squeeze(-1)

            # token_ids = torch.cat(
            #     (
            #         token_ids,
            #         next_token.unsqueeze(1),
            #     ),
            #     dim=1,
            # )
            #
            logits = logits[:, -1, :]

            # logits = self._apply_temperature(
            #     logits,
            #     temperature=temperature,
            # )

            # logits = self._apply_top_k(
            #     logits=logits,
            #     top_k=top_k,
            # )

            # logits = self._apply_top_p(
            #     logits=logits,
            #     top_p=top_p,
            # )

            logits = self._prepare_logits(
                logits=logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            next_token = self._sample_next_token(
                logits,
                temperature=temperature,
            )

            token_ids = torch.cat(
                (
                    token_ids,
                    next_token.unsqueeze(-1),
                ),
                dim=-1,
            )

        if was_training:
            self.train()

        return token_ids
