# TinyStory

A small GPT-style language model, written from scratch in PyTorch, that learns
to write short children's stories. The whole pipeline is included: data
download, BPE tokenizer training, a transformer implementation, a training
loop and text generation. It trains on a CPU, needs no external APIs, and the
code is meant to be read.

> **Status:** educational project. The default 4.2M-parameter model writes
> short, mostly grammatical stories but loses track of the plot over longer
> passages. See [MODEL_CARD.md](MODEL_CARD.md) for details and limitations.

## Features

- Decoder-only transformer (causal self-attention, pre-LayerNorm blocks, tied embeddings)
- Byte-level BPE tokenizer trained on your own corpus
- Data from [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) or
  [SimpleStories](https://huggingface.co/datasets/SimpleStories/SimpleStories),
  optionally mixed with public-domain books from Project Gutenberg
- Leak-free validation split (held-out stories and held-out book tails)
- Warm-start / continued training, best-checkpoint saving, optional JSON-lines logging
- Sampling with temperature, top-k and a repetition penalty
- Unit tests and a GitHub Actions workflow

## Installation

git clone https://github.com/NovumLabs/TinyStory.git
cd TinyStory
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

## Quick start

```bash
# 1. (Optional) download and clean public-domain books into books/
python download_books.py

# 2. Build the corpus, train the tokenizer and encode the data
python prepare_data.py --dataset tinystories --n-stories 50000

# 3. Train (about 4 hours on a consumer CPU for 8,000 iterations)
python train.py --config configs/cpu_4m.json

# 4. Generate
python generate.py "Once upon a time, a girl found a magic" --temperature 0.5

# 5. Or chat interactively
python chat.py
```

If `books/` is empty or missing, `prepare_data.py` simply trains on the story
dataset alone.

## Project structure

| Path | Purpose |
|------|---------|
| `config.py` | `ModelConfig` / `TrainConfig` dataclasses and JSON loading |
| `model.py` | Transformer, generation loop, checkpoint loading |
| `prepare_data.py` | Dataset download, splitting, tokenizer training, binary encoding |
| `train.py` | Training loop with cosine LR schedule and best-checkpoint saving |
| `generate.py` | Command-line text generation and the `generate_text` helper |
| `chat.py` | Interactive prompt loop |
| `download_books.py` | Download and clean Project Gutenberg books |
| `check_books.py` | Sanity-check text files before training |
| `configs/` | Ready-made configurations (`cpu_4m.json`, `gpu_14m.json`) |
| `examples/` | Example prompts and sampling tips |
| `tests/` | Unit tests (`pytest`) |
| `DATASET.md`, `MODEL_CARD.md`, `EXPERIMENTS.md` | Data sources, model description, experiment log |

## Usage details

### Preparing data

```bash
python prepare_data.py --dataset simplestories --n-stories 30000 --max-words 350
python prepare_data.py --vocab-size 8192 --book-repeat 2 --data-dir data_big
```

The script writes `train.bin`, `val.bin`, `tokenizer.json` and `meta.json` to
`--data-dir`. `train.py` reads the vocabulary size from `meta.json`, so the
model and tokenizer always match. Run `python check_books.py` first if you
add your own text files.

### Training

```bash
python train.py --config configs/cpu_4m.json
python train.py --max-iters 5000 --batch-size 16        # quicker run
python train.py --init-from ckpt.pt --lr 3e-4 --out ckpt_continued.pt
python train.py --log-file runs/log.jsonl               # metrics as JSON lines
```

Only the checkpoint with the best validation loss is written to `--out`
(default `ckpt.pt`). Press `Ctrl+C` at any time; the best checkpoint so far is
kept. `--init-from` loads weights only: the optimiser state and learning-rate
schedule start fresh.

Command-line options override values from the JSON config.

### Generating

```bash
python generate.py "One day, a little rabbit" --temperature 0.5 --num-samples 3
python generate.py "A fox met a crow" --rep-penalty 1.05 --seed 42
```

| Option | Meaning |
|--------|---------|
| `--temperature` | 0.5 - 0.6 works best for the small model; 0 is greedy |
| `--top-k` | Sample only from the k most likely tokens |
| `--rep-penalty` | Slightly above 1.0 (e.g. 1.05) reduces loops; larger values can produce merged words |
| `--max-new-tokens` | Output length; 150 - 180 is usually enough for a short story |

Complete sentences with a named character make better prompts than short
fragments.

## Results

| Run | Data | Params | Iterations | Best validation loss |
|-----|------|--------|-----------:|---------------------:|
| Default CPU run | 49.5K TinyStories + 3 Gutenberg books | 4.2M | 8,000 | **2.629** |

Validation loss is cross-entropy in nats per BPE token on this project's own
held-out split, so it cannot be compared directly with numbers from other
tokenizers or datasets. The full log, including failed experiments, is in
[EXPERIMENTS.md](EXPERIMENTS.md).

## Scaling up

`configs/gpu_14m.json` describes a ~14M-parameter model (6 layers, 384
dimensions, 256-token context). It has **not** been trained yet; treat it as a
starting point. Suggested workflow on a free Colab or Kaggle GPU:

```bash
python prepare_data.py --n-stories 200000 --vocab-size 8192
python train.py --config configs/gpu_14m.json
```

## Tests

```bash
pip install pytest
pytest -q
```

## Limitations

- Trained on synthetic children's stories: vocabulary and themes are narrow.
- The model repeats itself and mixes up characters in longer outputs.
- It continues a prompt; it does not follow instructions.
- English only.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgements

- Eldan & Li, [*TinyStories*](https://arxiv.org/abs/2305.07759), for the dataset and the idea that tiny models can write coherent stories.
- Finke et al., [*SimpleStories*](https://arxiv.org/abs/2504.09184).
- [Project Gutenberg](https://www.gutenberg.org) for public-domain texts.
- Andrej Karpathy's *nanoGPT* and "Let's build GPT" video, which inspired the structure of this code.

## Citation

If you use this model, dataset, or code in your research, please cite our technical report:

```bibtex
@techreport{khan2026tinystory,
  title={Developmental Analysis and Technical Report: Small Language Model (SLM) Training from Scratch},
  author={Khan, Arnab},
  institution={Novum Labs Pvt. Ltd.},
  year={2026},
  type={Preprint},
  month={October},
  version={v1.0.0},
  url={[https://zenodo.org/records/23223227](https://zenodo.org/records/23223227)}
}

## License

Code: [MIT](LICENSE). Datasets and books keep their own licenses; see
[DATASET.md](DATASET.md).
