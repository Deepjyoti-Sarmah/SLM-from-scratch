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

import torch

from src.layers.multi_head_attention import MultiHeadAttention

x = torch.randn(2, 4, 8)

attention = MultiHeadAttention(
    embedding_dim=8,
    num_heads=2,
    max_sequence_length=16,
)

output = attention(x)

print("\nReturned Output Shape:")
print(output.shape)
