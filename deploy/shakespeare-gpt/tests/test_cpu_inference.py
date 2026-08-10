import torch


def test_model_loads_on_cpu(model):
    assert next(model.parameters()).device.type == "cpu"


def test_cpu_generation_completes(model, tokenizer):
    prompt = "ROMEO:"

    token_ids = torch.tensor(
        tokenizer.encode(prompt),
        dtype=torch.long,
    ).unsqueeze(0)

    generated = model.generate(
        token_ids=token_ids,
        max_new_tokens=64,
        temperature=0.8,
        top_k=40,
    )

    text = tokenizer.decode(generated[0].tolist())

    assert len(text) == len(prompt) + 64
    assert text.startswith(prompt)