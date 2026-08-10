import torch


def test_generation_accepts_prompt(model, tokenizer):
    prompt = "ROMEO:"

    token_ids = torch.tensor(
        tokenizer.encode(prompt),
        dtype=torch.long,
    ).unsqueeze(0)

    generated = model.generate(
        token_ids=token_ids,
        max_new_tokens=32,
        temperature=0.0,
    )

    assert generated.shape[1] == len(prompt) + 32


def test_generated_output_is_valid_text(model, tokenizer):
    prompt = "ROMEO:"

    token_ids = torch.tensor(
        tokenizer.encode(prompt),
        dtype=torch.long,
    ).unsqueeze(0)

    generated = model.generate(
        token_ids=token_ids,
        max_new_tokens=32,
        temperature=0.0,
    )

    text = tokenizer.decode(generated[0].tolist())

    assert isinstance(text, str)
    assert text.startswith(prompt)
    assert len(text) == len(prompt) + 32


def test_generation_is_deterministic_with_seed(model, tokenizer):
    prompt = "Wherefore art thou"

    token_ids = torch.tensor(
        tokenizer.encode(prompt),
        dtype=torch.long,
    ).unsqueeze(0)

    torch.manual_seed(7)

    first = model.generate(
        token_ids=token_ids,
        max_new_tokens=32,
        temperature=0.8,
    )

    torch.manual_seed(7)

    second = model.generate(
        token_ids=token_ids,
        max_new_tokens=32,
        temperature=0.8,
    )

    assert torch.equal(first, second)