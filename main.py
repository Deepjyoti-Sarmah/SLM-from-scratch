# import torch

# from src.layers.self_attention import SelfAttention

# x = torch.randn(2, 4, 8)

# attention = SelfAttention(
#     embedding_dim=8,
#     max_sequence_length=16,
# )

# output = attention(x)

# print("Input Shape :", x.shape)
# print("Output Shape:", output.shape)

# print()

# print("===== Parameters =====")
# for name, param in attention.named_parameters():
#     print(name, param.shape)

# print()

# print("===== Buffers =====")
# for name, buffer in attention.named_buffers():
#     print(name, buffer.shape)

# print()

# print("Stored Mask:")
# print(attention.mask)

# print()

# T = x.size(1)

# print(f"Mask used for sequence length {T}:")
# print(attention.mask[:T, :T])

# import torch

# from src.layers.feed_forward import FeedForward
# from src.layers.multi_head_attention import MultiHeadAttention

# x = torch.randn(2, 4, 8)

# attention = MultiHeadAttention(
#     embedding_dim=8,
#     num_heads=2,
#     max_sequence_length=16,
# )

# output = attention(x)

# print("\nReturned Output Shape:")
# print(output.shape)

# x = torch.randn(2, 4, 8)

# feed_forward = FeedForward(
#     embedding_dim=8,
# )

# output = feed_forward(x)

# print("\nFinal Output:")
# print(output.shape)

import torch

from src.configs.gpt_config import GPTConfig
from src.models.gpt import GPT

config = GPTConfig(
    vocab_size=100,
    max_sequence_length=16,
    embedding_dim=64,
    num_heads=4,
    num_layers=2,
)

model = GPT(config=config)

token_ids = torch.randint(
    low=0,
    high=config.vocab_size,
    size=(2, 16),
)

logits = model(token_ids)

print(logits.shape)
