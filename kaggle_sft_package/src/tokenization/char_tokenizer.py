from __future__ import annotations

from collections.abc import Iterable


class CharacterTokenizer:
    def __init__(
        self,
        text: str,
    ) -> None:
        vocabulary = sorted(set(text))

        self.token_to_id = {
            token: token_id for token_id, token in enumerate(vocabulary)
        }

        self.id_to_token = {
            token_id: token for token_id, token in enumerate(vocabulary)
        }

    @property
    def vocab_size(
        self,
    ) -> int:
        return len(self.token_to_id)

    def encode(
        self,
        text: str,
    ) -> list[int]:
        return [self.token_to_id[token] for token in text]

    def decode(
        self,
        token_ids: Iterable[int],
    ) -> str:
        return "".join(self.id_to_token[token_id] for token_id in token_ids)
