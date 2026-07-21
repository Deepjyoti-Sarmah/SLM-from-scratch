from src.dataset import GPTDataset
from src.embedding import MyEmbedding
from src.tokenizer import CharacterTokenizer

from torch.utils.data import DataLoader


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

for inputs, targets in loader:
    print("Input IDs:")
    print(inputs)

    vectors = embedding(inputs)

    print("Embedding Shape:", vectors.shape)
    print(vectors)
    break
