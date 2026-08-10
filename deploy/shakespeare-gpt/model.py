"""Standalone GPT implementation for inference only.

This is a self-contained copy of the trained model architecture. It carries
no training code and matches the exact tensor names stored in
``model.safetensors``.

Expected weights (loaded with ``strict=True``) include keys such as:

    input_embedding.token_embedding.embedding.weight
    input_embedding.position_embedding.embedding.weight
    blocks.<i>.attention_norm.{weight,bias}
    blocks.<i>.multi_head_attention.qkv_projection.weight
    blocks.<i>.multi_head_attention.output_projection.weight
    blocks.<i>.feed_forward_norm.{weight,bias}
    blocks.<i>.feed_forward.input_projection.{weight,bias}
    blocks.<i>.feed_forward.output_projection.{weight,bias}
    final_norm.{weight,bias}
    lm_head.weight
"""

import torch
import torch.nn.functional as F
from torch import nn

from tokenizer import CharacterTokenizer

_LOGIT_MASK_VALUE = float("-inf")


class GPTConfig:
    """Configuration for the Shakespeare GPT model."""

    def __init__(
        self,
        *,
        vocab_size: int,
        max_sequence_length: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        dropout_probability: float = 0.0,
        bias: bool = False,
        initializer_range: float = 0.02,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_probability = dropout_probability
        self.bias = bias
        self.initializer_range = initializer_range

    @classmethod
    def from_json(
        cls,
        data: dict,
    ) -> "GPTConfig":
        return cls(
            vocab_size=data["vocab_size"],
            max_sequence_length=data["max_sequence_length"],
            embedding_dim=data["embedding_dim"],
            num_heads=data["num_heads"],
            num_layers=data["num_layers"],
            dropout_probability=data.get("dropout_probability", 0.0),
            bias=data.get("bias", False),
            initializer_range=data.get("initializer_range", 0.02),
        )

    def to_json(self) -> dict:
        return {
            "model_type": "shakespeare-gpt",
            "vocab_size": self.vocab_size,
            "max_sequence_length": self.max_sequence_length,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dropout_probability": self.dropout_probability,
            "bias": self.bias,
            "initializer_range": self.initializer_range,
        }


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


class PositionEmbedding(nn.Module):
    def __init__(
        self,
        max_sequence_length: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=max_sequence_length,
            embedding_dim=embedding_dim,
        )

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(position_ids)


class InputEmbedding(nn.Module):
    def __init__(
        self,
        *,
        config: GPTConfig,
    ) -> None:
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

        return self.dropout(input_embeddings)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        *,
        config: GPTConfig,
    ) -> None:
        super().__init__()

        if config.embedding_dim % config.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads

        self.qkv_projection = nn.Linear(
            in_features=config.embedding_dim,
            out_features=3 * config.embedding_dim,
            bias=config.bias,
        )

        self.output_projection = nn.Linear(
            in_features=config.embedding_dim,
            out_features=config.embedding_dim,
            bias=config.bias,
        )

        self.attention_dropout = nn.Dropout(config.dropout_probability)
        self.output_dropout = nn.Dropout(config.dropout_probability)

    def _split_heads(
        self,
        tensor: torch.Tensor,
        *,
        batch_size: int,
        sequence_length: int,
    ) -> torch.Tensor:
        return tensor.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        query, key, value = self.qkv_projection(hidden_states).split(
            self.embedding_dim,
            dim=-1,
        )

        query = self._split_heads(
            query,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        key = self._split_heads(
            key,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        value = self._split_heads(
            value,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )

        attention_output = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
            dropout_p=(self.attention_dropout.p if self.training else 0.0),
            is_causal=True,
        )

        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                self.embedding_dim,
            )
        )

        attention_output = self.output_projection(attention_output)

        return self.output_dropout(attention_output)


class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        config: GPTConfig,
    ) -> None:
        super().__init__()

        hidden_dim = 4 * config.embedding_dim

        self.input_projection = nn.Linear(
            in_features=config.embedding_dim,
            out_features=hidden_dim,
        )
        self.gelu = nn.GELU()
        self.output_projection = nn.Linear(
            in_features=hidden_dim,
            out_features=config.embedding_dim,
        )
        self.dropout = nn.Dropout(config.dropout_probability)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.input_projection(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.output_projection(hidden_states)

        return self.dropout(hidden_states)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        config: GPTConfig,
    ) -> None:
        super().__init__()

        self.attention_norm = nn.LayerNorm(normalized_shape=config.embedding_dim)
        self.multi_head_attention = MultiHeadAttention(config=config)

        self.feed_forward_norm = nn.LayerNorm(normalized_shape=config.embedding_dim)
        self.feed_forward = FeedForward(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.attention_norm(hidden_states)
        hidden_states = self.multi_head_attention(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.feed_forward_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GPT(nn.Module):
    """Generative pre-trained transformer for character-level text generation."""

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
            flattened_logits = logits.reshape(-1, self.vocab_size)
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

        k = min(top_k, self.vocab_size)

        values, _ = torch.topk(logits, k=k)

        threshold = values[:, -1].unsqueeze(-1)

        return logits.masked_fill(logits < threshold, _LOGIT_MASK_VALUE)

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

        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)

        cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

        sorted_mask = cumulative_probabilities > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(
            sorted_mask,
            _LOGIT_MASK_VALUE,
        )

        filtered_logits = torch.full_like(logits, _LOGIT_MASK_VALUE)

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
            return torch.argmax(logits, dim=-1)

        probabilities = torch.softmax(logits, dim=-1)

        return torch.multinomial(probabilities, num_samples=1).squeeze(-1)

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
        if temperature < 0:
            raise ValueError("temperature must be greater than or equal to 0.")

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if top_p is not None and not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in the range (0, 1].")

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than 0.")

        was_training = self.training

        self.eval()

        try:
            for _ in range(max_new_tokens):
                input_ids = token_ids[:, -self.config.max_sequence_length :]

                logits, _ = self(token_ids=input_ids)

                logits = logits[:, -1, :]

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
        finally:
            if was_training:
                self.train()

        return token_ids


def load_model_weights(
    model: GPT,
    path: str,
    *,
    device: torch.device | str = "cpu",
) -> None:
    from safetensors.torch import load_file

    state = load_file(
        path,
        device=str(device),
    )

    model.load_state_dict(
        state,
        strict=True,
    )


def parameter_count(
    model: GPT,
) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def generate_text(
    *,
    model: GPT,
    tokenizer: CharacterTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    device: torch.device | str,
) -> str:
    token_ids = tokenizer.encode(prompt)

    token_ids_tensor = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    generated = model.generate(
        token_ids=token_ids_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    return tokenizer.decode(generated[0].tolist())