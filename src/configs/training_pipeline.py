from dataclasses import dataclass

from src.models.gpt import GPT
from src.tokenization.char_tokenizer import CharacterTokenizer
from src.training.trainer import Trainer


@dataclass(slots=True)
class TrainingPipeline:
    trainer: Trainer
    model: GPT
    tokenizer: CharacterTokenizer
