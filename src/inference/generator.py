import torch

from src.models.gpt import GPT
from src.tokenization.char_tokenizer import CharacterTokenizer


class TextGenerator:
    def __init__(
        self,
        *,
        model: GPT,
        tokenizer: CharacterTokenizer,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> str:
        token_ids = self.tokenizer.encode(prompt)

        token_ids_tensor = torch.tensor(
            token_ids,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

        generated = self.model.generate(
            token_ids=token_ids_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        text = self.tokenizer.decode(generated[0].tolist())

        return text
