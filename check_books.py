from __future__ import annotations

import argparse
import glob
import os
import statistics
from typing import Dict, List, Optional

from tokenizers import Tokenizer

#: Thresholds used to raise warnings.
MIN_CHARS_PER_TOKEN = 2.0
MAX_NON_ASCII_RATIO = 0.10
HARD_WRAP_LINE_RANGE = (40, 80)
HARD_WRAP_RATIO = 0.6


def analyze_text(text: str, tokenizer: Optional[Tokenizer] = None) -> Dict:
    """Return summary statistics and warnings for a text.

    Args:
        text: Full text of a book.
        tokenizer: Optional tokenizer used to compute tokens and chars/token.
    """
    n_chars = len(text)
    non_ascii = sum(1 for c in text if ord(c) > 127) / max(n_chars, 1)

    lines = [ln for ln in text.splitlines() if ln.strip()]
    lo, hi = HARD_WRAP_LINE_RANGE
    wrapped = sum(1 for ln in lines if lo <= len(ln) <= hi) / max(len(lines), 1)
    median_line = statistics.median(len(ln) for ln in lines) if lines else 0

    stats: Dict = {
        "chars": n_chars,
        "non_ascii_ratio": non_ascii,
        "median_line_length": median_line,
        "suspected_hard_wrap": wrapped > HARD_WRAP_RATIO and len(lines) > 20,
        "tokens": None,
        "chars_per_token": None,
        "warnings": [],
    }
    if tokenizer is not None and n_chars:
        n_tokens = len(tokenizer.encode(text).ids)
        stats["tokens"] = n_tokens
        stats["chars_per_token"] = n_chars / max(n_tokens, 1)
        if stats["chars_per_token"] < MIN_CHARS_PER_TOKEN:
            stats["warnings"].append(
                "Very low chars/token: the text may not be English, or the "
                "tokenizer was trained on different data."
            )
    if non_ascii > MAX_NON_ASCII_RATIO:
        stats["warnings"].append("High share of non-ASCII characters.")
    if stats["suspected_hard_wrap"]:
        stats["warnings"].append(
            "Lines look hard-wrapped; join lines within paragraphs "
            "(download_books.py does this automatically)."
        )
    return stats


def excerpt(text: str, length: int = 300) -> str:
    """Return ``length`` characters from the middle of ``text`` on one line."""
    mid = len(text) // 2
    return text[mid:mid + length].replace("\n", " ")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect plain-text training books.")
    p.add_argument("--books-dir", default="books")
    p.add_argument("--tokenizer", default="data/tokenizer.json",
                   help="Optional; token statistics are skipped if missing.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = (
        Tokenizer.from_file(args.tokenizer) if os.path.exists(args.tokenizer) else None
    )
    if tokenizer is None:
        print(f"Tokenizer '{args.tokenizer}' not found; skipping token statistics.")

    paths: List[str] = sorted(glob.glob(os.path.join(args.books_dir, "*.txt")))
    if not paths:
        print(f"No .txt files found in '{args.books_dir}'.")
        return

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        s = analyze_text(text, tokenizer)
        print(f"\n=== {path} ===")
        line = f"chars: {s['chars']:,} | non-ASCII: {s['non_ascii_ratio']:.1%}"
        if s["tokens"] is not None:
            line += f" | tokens: {s['tokens']:,} | chars/token: {s['chars_per_token']:.2f}"
        print(line)
        for warning in s["warnings"]:
            print(f"  WARNING: {warning}")
        print(f"  excerpt: {excerpt(text)}")


if __name__ == "__main__":
    main()
