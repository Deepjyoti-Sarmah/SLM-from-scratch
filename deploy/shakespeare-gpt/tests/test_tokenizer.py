def test_vocab_size(tokenizer):
    assert tokenizer.vocab_size == 65


def test_encode_decode_roundtrip(tokenizer):
    text = "ROMEO:"

    ids = tokenizer.encode(text)

    decoded = tokenizer.decode(ids)

    assert decoded == text


def test_vocabulary_is_stable(tokenizer):
    for token_id in range(65):
        token = tokenizer.id_to_token[token_id]
        assert tokenizer.token_to_id[token] == token_id