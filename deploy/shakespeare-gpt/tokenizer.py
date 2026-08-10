"""Character-level tokenizer matching the trained vocabulary exactly.

Loads the ``tokenizer.json`` artifact produced during training. The
token-to-id mapping is preserved verbatim; it is never re-derived from a
text file, so token ids stay stable.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path


class CharacterTokenizer:
    def __init__(
        self,
        *,
        token_to_id: dict[str, int],
        id_to_token: dict[int, str],
    ) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token

    @classmethod
    def from_file(
        cls,
        path: str | Path,
    ) -> "CharacterTokenizer":
        data = json.loads(
            Path(path).read_text(
                encoding="utf-8",
            )
        )

        token_to_id = {
            token: int(token_id)
            for token, token_id in data["token_to_id"].items()
        }

        id_to_token = {
            int(token_id): token
            for token_id, token in data["id_to_token"].items()
        }

        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
        )

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