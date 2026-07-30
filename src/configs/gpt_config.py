from dataclasses import dataclass


@dataclass(slots=True)
class GPTConfig:
    vocab_size: int
    max_sequence_length: int

    embedding_dim: int
    num_heads: int
    num_layers: int

    dropout_probability: float = 0.1

    bias: bool = False
