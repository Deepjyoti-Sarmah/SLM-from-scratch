import torch
from torch.utils.data import DataLoader

from src.embeddings.token_embedding import MyEmbedding
from src.layers.self_attention import SelfAttention
from src.tokenization.dataset import GPTDataset
from src.tokenization.tokenizer import CharacterTokenizer

# text = "banana"

# tokenizer = CharacterTokenizer(text)

# tokens = tokenizer.encode(text)

# print(tokens)

# dataset = GPTDataset(
#     tokens=tokens,
#     context_size=3,
# )

# print(len(dataset))


# loader = DataLoader(
#     dataset,
#     batch_size=2,
#     shuffle=False,
# )

# embedding = MyEmbedding(
#     vocab_size=len(tokenizer.chars),
#     embedding_dim=4,
# )

# print(embedding.embedding.weight.shape)
# print(embedding.embedding.weight)

# for inputs, targets in loader:
#     print("Input IDs:")
#     # print(inputs)

#     vectors = embedding(inputs)

#     print("Embedding Shape:", vectors.shape)
#     print(vectors)
#     break

# position_ids = torch.arange(3)

# print(position_ids)
# print(type(position_ids))
# print(position_ids.shape)

x = torch.randn(2, 4, 512)

attention = SelfAttention(embedding_dim=512)

scores, value = attention(x)

query = attention.query(x)
key = attention.key(x)


print("Q shape:", query.shape)
print("K shape:", key.shape)
print("Scores shape:", scores.shape)

print(scores.min())
print(scores.max())
print(scores.mean())
print(scores.std())
