import sys
from pathlib import Path

import json
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def tokenizer():
    from tokenizer import CharacterTokenizer

    return CharacterTokenizer.from_file(REPO_ROOT / "tokenizer.json")


@pytest.fixture(scope="session")
def config_data():
    return json.loads(
        (REPO_ROOT / "config.json").read_text(
            encoding="utf-8",
        )
    )


@pytest.fixture(scope="session")
def model(config_data):
    from safetensors.torch import load_file

    from model import GPT, GPTConfig

    config = GPTConfig.from_json(config_data)

    model = GPT(config=config)

    state = load_file(
        REPO_ROOT / "model.safetensors",
        device="cpu",
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()

    return model