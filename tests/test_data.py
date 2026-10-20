"""Tests for data preparation, book cleaning and book analysis."""

import json

import numpy as np
import pytest

from check_books import analyze_text
from config import EOT_TOKEN
from download_books import clean_gutenberg_text
from prepare_data import (
    build_splits,
    encode_documents,
    load_books,
    split_books,
    write_corpus,
)
from train import get_batch, load_split


def test_split_books_keeps_head_and_tail():
    train, val = split_books(["a" * 100], val_fraction=0.1)
    assert len(train[0]) == 90 and len(val[0]) == 10


def test_build_splits_holds_out_validation_and_repeats_books():
    stories = [f"story {i}" for i in range(20)]
    books = ["B" * 100]
    train, val = build_splits(stories, books, n_val_stories=3, book_repeat=2,
                              book_val_fraction=0.1)
    assert [s for s in train if s.startswith("story")] == stories[:-3]
    assert val[:3] == stories[-3:]
    assert train.count("B" * 90) == 2           # repeated in train only
    assert val[-1] == "B" * 10                  # held-out tail appears once
    assert not set(stories[-3:]) & set(train)   # no leakage


def test_build_splits_requires_enough_stories():
    with pytest.raises(ValueError):
        build_splits(["a", "b"], [], n_val_stories=5)


def test_load_books_reads_txt_files_sorted(tmp_path):
    (tmp_path / "b.txt").write_text("second", encoding="utf-8")
    (tmp_path / "a.txt").write_text("first", encoding="utf-8")
    (tmp_path / "ignore.md").write_text("nope", encoding="utf-8")
    assert load_books(str(tmp_path)) == ["first", "second"]


def test_tokenizer_has_eos_and_encoding_appends_it(tiny_tokenizer):
    eos_id = tiny_tokenizer.token_to_id(EOT_TOKEN)
    assert eos_id is not None
    docs = ["Once upon a time.", "A boy found a ball."]
    ids = encode_documents(tiny_tokenizer, docs, eos_id)
    assert ids.dtype == np.uint16
    assert int((ids == eos_id).sum()) == len(docs)
    assert ids[-1] == eos_id


def test_write_corpus_and_binary_roundtrip(tmp_path, tiny_tokenizer):
    docs = ["Once upon a time.", "A boy found a ball."]
    corpus = tmp_path / "corpus.txt"
    write_corpus(docs, str(corpus))
    assert EOT_TOKEN in corpus.read_text(encoding="utf-8")

    eos_id = tiny_tokenizer.token_to_id(EOT_TOKEN)
    ids = encode_documents(tiny_tokenizer, docs * 20, eos_id)
    ids.tofile(tmp_path / "train.bin")
    ids.tofile(tmp_path / "val.bin")
    (tmp_path / "meta.json").write_text(json.dumps({"vocab_size": 300}))

    data = load_split(str(tmp_path), "train")
    assert len(data) == len(ids)

    import torch

    x, y = get_batch(data, 4, 8, torch.device("cpu"), np.random.default_rng(0))
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert torch.equal(x[:, 1:], y[:, :-1])     # targets are inputs shifted by one


def test_load_split_missing_file_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_data.py"):
        load_split(str(tmp_path), "train")


def test_clean_gutenberg_text():
    raw = (
        "Header and license text\r\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\r\n\r\n"
        "[Illustration: a picture]\r\n\r\n"
        "\u201cHello,\u201d said the fox. It\u2019s a fine\r\n"
        "day for a walk.\r\n\r\n"
        "A second paragraph.\r\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\r\n"
        "Footer and license text\r\n"
    )
    cleaned = clean_gutenberg_text(raw)
    assert "license" not in cleaned.lower()
    assert "Illustration" not in cleaned
    assert '"Hello," said the fox. It\'s a fine day for a walk.' in cleaned
    assert cleaned.split("\n\n") == [
        '"Hello," said the fox. It\'s a fine day for a walk.',
        "A second paragraph.",
    ]


def test_clean_gutenberg_text_without_markers_keeps_content():
    assert clean_gutenberg_text("Just some text.") == "Just some text."


def test_analyze_text_flags_hard_wrapping_and_non_ascii(tiny_tokenizer):
    wrapped = "\n".join(["this line is wrapped at about sixty characters wide ok"] * 40)
    stats = analyze_text(wrapped, tiny_tokenizer)
    assert stats["suspected_hard_wrap"]
    assert stats["tokens"] and stats["chars_per_token"] >= 1

    foreign = "\u0986\u09ae\u09bf " * 200
    stats = analyze_text(foreign)
    assert stats["non_ascii_ratio"] > 0.5
    assert any("non-ASCII" in w for w in stats["warnings"])


def test_analyze_text_handles_empty_input():
    stats = analyze_text("")
    assert stats["chars"] == 0 and stats["warnings"] == []
