class CharacterTokenizer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.chars = sorted(set(text))

        self.stoi = {ch: index for index, ch in enumerate(self.chars)}

        self.iots = {index: ch for index, ch in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        ids = []

        for ch in text:
            ids.append(self.stoi[ch])

        return ids

    def decode(self, ids: list[int]) -> str:
        chars = []

        for token_id in ids:
            chars.append(self.iots[token_id])

        return "".join(chars)
