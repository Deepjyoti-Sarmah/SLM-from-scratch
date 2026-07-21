from src.tokenizer import CharacterTokenizer

tokenizer = CharacterTokenizer("banana")

encoded = tokenizer.encode("banana")

print(encoded)

decoded = tokenizer.decode(encoded)

print(decoded)


# chars = ["a", "b", "c"]
#
# print(enumerate(chars))
#
# for item in enumerate(chars):
#     print(item)
#
# for idx, ch in enumerate(chars):
#     print(idx, ch)
