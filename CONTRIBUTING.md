# Contributing

Thanks for your interest in improving TinyStory! Bug reports, documentation
fixes and focused improvements are all welcome.

## Development setup

```bash
git clone https://github.com/<your-username>/TinyStory.git
cd TinyStory
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt pytest
pytest -q
```

## Guidelines

- Keep the code readable and educational; prefer clarity over cleverness.
- Add type hints and short docstrings to public functions.
- Add or update tests for any behaviour change (`tests/`). Tests must run
  offline and on a CPU in a few seconds: use tiny models and tiny tokenizers,
  never real datasets.
- Do not commit data, checkpoints or books (`data/`, `books/`, `*.pt`, `*.bin`
  are ignored). Do not add copyrighted text.
- Record notable experiments, including negative results, in `EXPERIMENTS.md`
  with the exact configuration so that others can reproduce them.

## Pull requests

1. Open an issue first for larger changes so the direction can be discussed.
2. Create a branch, make your change and run `pytest`.
3. Describe what changed and why; include before/after numbers for anything
   that affects training or generation quality.
4. Make sure the GitHub Actions workflow passes.

## Reporting bugs

Please include your Python and PyTorch versions, the command you ran, the full
error message and, for training problems, the first lines of the log output.
