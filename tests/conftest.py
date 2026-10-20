"""Shared pytest fixtures."""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ModelConfig  # noqa: E402
from model import GPT  # noqa: E402
from prepare_data import train_tokenizer  # noqa: E402

SAMPLE_STORIES = [
    "Once upon a time, a little girl named Lily went to the park. She saw a big dog.",
    "One day, a boy named Tim found a red ball. He wanted to play with his friend.",
    "A clever fox met a wise crow. The crow said, \"Hello!\" and they became friends.",
    "Mia had a cat named Tom. They played in the garden all day and had fun.",
] * 25


@pytest.fixture
def tiny_config() -> ModelConfig:
    return ModelConfig(vocab_size=300, block_size=16, n_layer=2, n_head=2,
                       n_embd=32, dropout=0.0)


@pytest.fixture
def tiny_model(tiny_config) -> GPT:
    import torch

    torch.manual_seed(0)
    return GPT(tiny_config).eval()


@pytest.fixture
def tiny_tokenizer(tmp_path):
    """A small BPE tokenizer (vocab <= 300) trained on a few sentences."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n<|endoftext|>\n".join(SAMPLE_STORIES), encoding="utf-8")
    return train_tokenizer(str(corpus), 300, str(tmp_path / "tokenizer.json"))
