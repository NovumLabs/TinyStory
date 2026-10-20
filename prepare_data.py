from __future__ import annotations

import argparse
import glob
import json
import os
from itertools import islice
from typing import List, Optional, Sequence, Tuple

import numpy as np
from tokenizers import ByteLevelBPETokenizer, Tokenizer

from config import EOT_TOKEN


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_tinystories(n_stories: int) -> List[str]:
    """Load the first ``n_stories`` stories of TinyStories from Hugging Face."""
    from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories", split=f"train[:{n_stories}]")
    stories = [x["text"].strip() for x in ds]
    return [s for s in stories if s]


def load_simplestories(
    n_stories: int, max_words: Optional[int] = 350, scan_limit: int = 200_000
) -> List[str]:
    """Stream SimpleStories and keep up to ``n_stories`` short stories.

    Streaming avoids downloading the full ~2 GB dataset. Stories longer than
    ``max_words`` are skipped because very small models struggle with them.
    """
    from datasets import load_dataset

    ds = load_dataset("SimpleStories/SimpleStories", split="train", streaming=True)
    stories: List[str] = []
    for row in islice(ds, scan_limit):
        text = row["story"].strip()
        if not text or text.startswith("---"):
            continue
        if max_words is not None and row["word_count"] > max_words:
            continue
        stories.append(text)
        if len(stories) >= n_stories:
            break
    return stories


def load_books(books_dir: str) -> List[str]:
    """Read every ``*.txt`` file in ``books_dir`` (UTF-8)."""
    books = []
    for path in sorted(glob.glob(os.path.join(books_dir, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            books.append(f.read())
    return books


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def split_books(
    books: Sequence[str], val_fraction: float = 0.05
) -> Tuple[List[str], List[str]]:
    """Split each book into a training head and a held-out tail."""
    train_parts, val_parts = [], []
    for text in books:
        cut = int(len(text) * (1.0 - val_fraction))
        train_parts.append(text[:cut])
        val_parts.append(text[cut:])
    return train_parts, val_parts


def build_splits(
    stories: Sequence[str],
    books: Sequence[str],
    n_val_stories: int = 500,
    book_repeat: int = 3,
    book_val_fraction: float = 0.05,
) -> Tuple[List[str], List[str]]:
    """Create train/validation document lists.

    The last ``n_val_stories`` stories and the last ``book_val_fraction`` of
    every book are held out. Books are repeated ``book_repeat`` times in the
    *training* split only, so validation text is never seen during training.
    """
    if len(stories) <= 2 * n_val_stories:
        raise ValueError(
            f"Need more than {2 * n_val_stories} stories, got {len(stories)}."
        )
    train_stories = list(stories[:-n_val_stories])
    val_stories = list(stories[-n_val_stories:])
    train_books, val_books = split_books(books, book_val_fraction)
    train_docs = train_stories + train_books * book_repeat
    val_docs = val_stories + val_books
    return train_docs, val_docs


# --------------------------------------------------------------------------- #
# Tokenizer and encoding
# --------------------------------------------------------------------------- #
def write_corpus(docs: Sequence[str], path: str) -> None:
    """Write documents separated by the end-of-text token."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"\n{EOT_TOKEN}\n".join(docs))


def train_tokenizer(corpus_path: str, vocab_size: int, save_path: str) -> Tokenizer:
    """Train a byte-level BPE tokenizer and save it as JSON."""
    bpe = ByteLevelBPETokenizer()
    bpe.train(
        files=[corpus_path],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[EOT_TOKEN],
    )
    bpe.save(save_path)
    return Tokenizer.from_file(save_path)


def encode_documents(
    tokenizer: Tokenizer, docs: Sequence[str], eos_id: int, batch_size: int = 1000
) -> np.ndarray:
    """Encode documents and append ``eos_id`` after each one."""
    ids: List[int] = []
    for start in range(0, len(docs), batch_size):
        for enc in tokenizer.encode_batch(list(docs[start:start + batch_size])):
            ids.extend(enc.ids)
            ids.append(eos_id)
    return np.array(ids, dtype=np.uint16)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", choices=["tinystories", "simplestories"],
                   default="tinystories", help="Story dataset to download.")
    p.add_argument("--n-stories", type=int, default=50_000,
                   help="Number of stories to use (including validation stories).")
    p.add_argument("--n-val-stories", type=int, default=500,
                   help="Stories held out for validation.")
    p.add_argument("--max-words", type=int, default=350,
                   help="SimpleStories only: skip stories longer than this.")
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--book-repeat", type=int, default=3,
                   help="Times each book is repeated in the training split.")
    p.add_argument("--books-dir", default="books")
    p.add_argument("--data-dir", default="data")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    print(f"Loading {args.dataset} ...")
    if args.dataset == "tinystories":
        stories = load_tinystories(args.n_stories)
    else:
        stories = load_simplestories(args.n_stories, args.max_words)
    print(f"Loaded {len(stories):,} stories")

    books = load_books(args.books_dir)
    print(f"Loaded {len(books)} book(s) from '{args.books_dir}'")

    train_docs, val_docs = build_splits(
        stories, books, args.n_val_stories, args.book_repeat
    )

    corpus_path = os.path.join(args.data_dir, "corpus.txt")
    tokenizer_path = os.path.join(args.data_dir, "tokenizer.json")
    write_corpus(train_docs, corpus_path)

    print("Training BPE tokenizer ...")
    tokenizer = train_tokenizer(corpus_path, args.vocab_size, tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size > np.iinfo(np.uint16).max:
        raise ValueError("Vocabulary too large for uint16 storage.")
    eos_id = tokenizer.token_to_id(EOT_TOKEN)

    train_ids = encode_documents(tokenizer, train_docs, eos_id)
    val_ids = encode_documents(tokenizer, val_docs, eos_id)
    train_ids.tofile(os.path.join(args.data_dir, "train.bin"))
    val_ids.tofile(os.path.join(args.data_dir, "val.bin"))

    meta = {
        "dataset": args.dataset,
        "n_stories": len(stories),
        "n_val_stories": args.n_val_stories,
        "n_books": len(books),
        "book_repeat": args.book_repeat,
        "vocab_size": vocab_size,
        "eos_id": eos_id,
        "train_tokens": int(len(train_ids)),
        "val_tokens": int(len(val_ids)),
    }
    with open(os.path.join(args.data_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Vocabulary size: {vocab_size}")
    print(f"Train tokens: {len(train_ids):,} | Val tokens: {len(val_ids):,}")


if __name__ == "__main__":
    main()
