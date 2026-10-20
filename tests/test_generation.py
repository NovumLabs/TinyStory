"""Tests for the text-generation helper."""

from generate import generate_text


def test_returns_string_starting_with_prompt(tiny_model, tiny_tokenizer):
    prompt = "Once upon a time"
    text = generate_text(tiny_model, tiny_tokenizer, prompt, max_new_tokens=5, seed=0)
    assert isinstance(text, str)
    assert text.startswith(prompt)


def test_same_seed_gives_same_output(tiny_model, tiny_tokenizer):
    kwargs = dict(max_new_tokens=12, temperature=1.0, seed=123)
    a = generate_text(tiny_model, tiny_tokenizer, "One day", **kwargs)
    b = generate_text(tiny_model, tiny_tokenizer, "One day", **kwargs)
    assert a == b


def test_greedy_decoding_ignores_seed(tiny_model, tiny_tokenizer):
    a = generate_text(tiny_model, tiny_tokenizer, "One day", max_new_tokens=8,
                      temperature=0, seed=1)
    b = generate_text(tiny_model, tiny_tokenizer, "One day", max_new_tokens=8,
                      temperature=0, seed=2)
    assert a == b


def test_empty_prompt_is_supported(tiny_model, tiny_tokenizer):
    text = generate_text(tiny_model, tiny_tokenizer, "", max_new_tokens=5, seed=0)
    assert isinstance(text, str)


def test_repetition_penalty_option(tiny_model, tiny_tokenizer):
    text = generate_text(tiny_model, tiny_tokenizer, "A fox", max_new_tokens=10,
                         rep_penalty=1.1, rep_window=8, seed=0)
    assert isinstance(text, str)


def test_zero_new_tokens_returns_prompt(tiny_model, tiny_tokenizer):
    text = generate_text(tiny_model, tiny_tokenizer, "Hello there", max_new_tokens=0)
    assert text == "Hello there"
