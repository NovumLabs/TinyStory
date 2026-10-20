from __future__ import annotations

import argparse
import os
import re
import time
import urllib.request
from typing import Dict

#: Default books: name -> Project Gutenberg ebook id.
DEFAULT_BOOKS: Dict[str, int] = {
    "grimm": 52521,           # Grimm's Fairy Tales
    "andersen": 66688,        # Hans Andersen's Fairy Tales
    "aesop_children": 19994,  # The Aesop for Children
}

GUTENBERG_URL = "https://www.gutenberg.org/ebooks/{id}.txt.utf-8"
USER_AGENT = "Mozilla/5.0 (compatible; TinyStory-downloader)"


def clean_gutenberg_text(raw: str) -> str:
    """Convert a raw Project Gutenberg text file into clean prose.

    Steps: keep only the text between the START/END markers, drop illustration
    notes, normalise curly quotes to ASCII, and join hard-wrapped lines so that
    each paragraph becomes a single line (paragraphs are separated by a blank
    line).
    """
    text = raw.replace("\r\n", "\n")
    match = re.search(r"\*\*\* START OF .*?\*\*\*(.*?)\*\*\* END OF", text, re.S)
    if match:
        text = match.group(1)
    text = re.sub(r"\[Illustration[^\]]*\]", "", text)
    text = (text.replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2018", "'").replace("\u2019", "'"))
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    return "\n\n".join(p for p in paragraphs if p)


def download_book(book_id: int, timeout: int = 60) -> str:
    """Download the plain-text UTF-8 version of a Gutenberg ebook."""
    request = urllib.request.Request(
        GUTENBERG_URL.format(id=book_id), headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download public-domain books.")
    p.add_argument("--book", action="append", metavar="NAME=ID",
                   help="Book to download (repeatable). Defaults to a small "
                        "set of fairy tales and fables.")
    p.add_argument("--raw-dir", default="raw_books")
    p.add_argument("--out-dir", default="books")
    p.add_argument("--delay", type=float, default=1.0,
                   help="Seconds to wait between downloads.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    books = dict(DEFAULT_BOOKS)
    if args.book:
        books = {}
        for item in args.book:
            name, _, book_id = item.partition("=")
            if not name or not book_id.isdigit():
                raise SystemExit(f"Invalid --book value '{item}'; use NAME=ID.")
            books[name] = int(book_id)

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    for i, (name, book_id) in enumerate(books.items()):
        if i:
            time.sleep(args.delay)
        print(f"Downloading '{name}' (Gutenberg #{book_id}) ...")
        try:
            raw = download_book(book_id)
        except Exception as exc:  # network errors, HTTP errors, decoding errors
            print(f"  Failed: {exc}\n  Download it manually from "
                  f"https://www.gutenberg.org/ebooks/{book_id} "
                  f"(Plain Text UTF-8) and save it to {args.raw_dir}/{name}.txt")
            continue
        cleaned = clean_gutenberg_text(raw)
        with open(os.path.join(args.raw_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write(raw)
        with open(os.path.join(args.out_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write(cleaned)
        print(f"  raw {len(raw):,} chars -> cleaned {len(cleaned):,} chars")


if __name__ == "__main__":
    main()
