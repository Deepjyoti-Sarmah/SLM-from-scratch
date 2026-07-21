class GPTDataset:
    def __init__(
        self,
        tokens: list[int],
        context_size: int,
    ):
        self.tokens = tokens
        self.context_size = context_size

    def __len__(self) -> int:
        n = len(self.tokens)
        c = self.context_size

        return n - c

    def __getitem__(self, index: int):
        input_ids = self.tokens[index : index + self.context_size]
        target_ids = self.tokens[index + 1 : index + self.context_size + 1]

        return input_ids, target_ids
