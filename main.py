import torch
from torch.utils.data import DataLoader

from src.embeddings.token_embedding import MyEmbedding
from src.tokenization.dataset import GPTDataset
from src.tokenization.tokenizer import CharacterTokenizer

text = "banana"

tokenizer = CharacterTokenizer(text)

tokens = tokenizer.encode(text)

print(tokens)

dataset = GPTDataset(
    tokens=tokens,
    context_size=3,
)

print(len(dataset))


loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
)

embedding = MyEmbedding(
    vocab_size=len(tokenizer.chars),
    embedding_dim=4,
)

print(embedding.embedding.weight.shape)
print(embedding.embedding.weight)

for inputs, targets in loader:
    print("Input IDs:")
    # print(inputs)

    vectors = embedding(inputs)

    print("Embedding Shape:", vectors.shape)
    print(vectors)
    break

position_ids = torch.arange(3)

print(position_ids)
print(type(position_ids))
print(position_ids.shape)
