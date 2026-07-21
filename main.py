from src.dataset import GPTDataset
from src.tokenizer import CharacterTokenizer

# tokenizer = CharacterTokenizer("banana")
#
# encoded = tokenizer.encode("banana")
#
# print(encoded)
#
# decoded = tokenizer.decode(encoded)
#
# None print(decoded)

tokens = [10, 20, 30, 40, 50, 60, 70]

dataset = GPTDataset(tokens, context_size=3)

print(len(dataset))

for i in range(len(dataset)):
    x, y = dataset[i]
    print(f"{i=}")
    print("input :", x)
    print("output :", y)
    print()


# chars = ["a", "b", "c"]
#
# print(enumerate(chars))
#
# for item in enumerate(chars):
#     print(item)
#
# for idx, ch in enumerate(chars):
#     print(idx, ch)
