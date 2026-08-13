from pathlib import Path

from safetensors.torch import load_file

from model import GPT, GPTConfig

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PACKAGE_ROOT / "model.safetensors"

EXPECTED_PARAMETER_COUNT = 4_782_336


def test_state_dict_keys_match(config_data, model):
    state_names = set(model.state_dict().keys())

    loaded = load_file(
        MODEL_PATH,
        device="cpu",
    )

    assert set(loaded.keys()) == state_names


def test_strict_load_succeeds(config_data, model):
    loaded = load_file(
        MODEL_PATH,
        device="cpu",
    )

    result = GPT(config=GPTConfig.from_json(config_data)).load_state_dict(
        loaded,
        strict=True,
    )

    assert not result.missing_keys
    assert not result.unexpected_keys


def test_parameter_count(model):
    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert parameters == EXPECTED_PARAMETER_COUNT