"""Step 6 verification: weights must load with strict=True and match the expected parameter count."""

import json
from pathlib import Path

from safetensors.torch import load_file

from model import GPT, GPTConfig, parameter_count

CONFIG_PATH = Path("config.json")
WEIGHTS_PATH = Path("model.safetensors")


def main() -> None:
    config_data = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8",
        )
    )

    config = GPTConfig.from_json(config_data)

    model = GPT(config=config)

    state = load_file(
        WEIGHTS_PATH,
        device="cpu",
    )

    result = model.load_state_dict(
        state,
        strict=True,
    )

    print(result)

    parameters = parameter_count(model)

    print(f"Parameters: {parameters:,}")

    assert parameters == 4_782_336


if __name__ == "__main__":
    main()