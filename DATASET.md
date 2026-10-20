# Datasets

This repository does not ship any data. `prepare_data.py` and
`download_books.py` fetch everything at run time. Always check the license
terms of each source before redistributing data or trained weights.

## Story datasets (Hugging Face)

### TinyStories

- Hub: https://huggingface.co/datasets/roneneldan/TinyStories
- Paper: Eldan & Li, *TinyStories: How Small Can Language Models Be and Still
  Speak Coherent English?* (2023), https://arxiv.org/abs/2305.07759
- Synthetic short stories written with a limited vocabulary typical of young
  children. Used by default (`--dataset tinystories`, first N stories of the
  `train` split).

### SimpleStories

- Hub: https://huggingface.co/datasets/SimpleStories/SimpleStories
- Paper: Finke et al., *Parameterized Synthetic Text Generation with
  SimpleStories* (2025), https://arxiv.org/abs/2504.09184
- More than two million synthetic stories with greater thematic and stylistic
  variety. English stories are ASCII-only. The `story` column holds the text;
  `word_count` is used to skip long stories. Loaded with streaming so the full
  ~2 GB download is not required. Selected with `--dataset simplestories`.
- The dataset card lists the MIT license.

## Books (Project Gutenberg)

`download_books.py` downloads these public-domain collections by default:

| Name | Title | Gutenberg ID |
|------|-------|-------------:|
| `grimm` | Grimm's Fairy Tales | 52521 |
| `andersen` | Hans Andersen's Fairy Tales | 66688 |
| `aesop_children` | The Aesop for Children | 19994 |

The texts are in the public domain in the United States; verify the status in
your own jurisdiction. The cleaning step removes the Project Gutenberg header
and footer, so if you redistribute the cleaned files you must follow the
[Project Gutenberg license terms](https://www.gutenberg.org/policy/license.html).
Other titles can be added with `--book NAME=ID`.

Do not add copyrighted books to a repository or to a model you intend to
publish unless you hold the rights to do so.

## Preparation pipeline

1. Load N stories and any `books/*.txt` files.
2. Hold out validation data:
   - the last 500 stories (`--n-val-stories`), and
   - the last 5% of every book.
3. Build the training split: remaining stories plus each book's first 95%,
   repeated `--book-repeat` times (default 3) so the small book collection is
   not drowned out by the larger story set.
4. Train a byte-level BPE tokenizer (default vocabulary 4,096) on the training
   text only. The special token `<|endoftext|>` separates documents.
5. Encode both splits to `uint16` arrays, appending the end-of-text token after
   each document, and write `meta.json`.

Because validation text is never repeated or used for tokenizer training,
validation loss is a fair estimate of generalisation.

## Output files

| File | Description |
|------|-------------|
| `data/corpus.txt` | Training text used to fit the tokenizer |
| `data/tokenizer.json` | Trained BPE tokenizer |
| `data/train.bin`, `data/val.bin` | Token ids, `uint16` |
| `data/meta.json` | Vocabulary size, token counts, dataset provenance |

## Using your own text

Place UTF-8 `.txt` files in `books/` and run `python check_books.py`. It
warns about hard-wrapped lines, OCR noise (low characters-per-token ratio) and
high non-ASCII content. Poorly scanned books hurt small models; during this
project one OCR-garbled collection was removed from the training data.
