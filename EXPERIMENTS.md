# Experiment Log

All runs used a single consumer CPU (no GPU), batch size 32, context length
128 and the default 4.2M-parameter model (4 layers, 4 heads, 256 dimensions,
4,096-token vocabulary) unless stated otherwise. Losses are cross-entropy in
nats per token.

> Validation sets differ between runs (see the notes), so validation losses
> are **not** directly comparable across runs. Compare generated samples
> instead.

## Summary

| Run | Training data | Iterations | Final train | Best val | Wall time | Outcome |
|-----|---------------|-----------:|------------:|---------:|----------:|---------|
| R1 | 20K TinyStories + 4 books x5 (12.2M tokens) | 2,250 (stopped) | 4.810 | 5.393 | ~59 min | Discarded: init bug and leaky validation |
| R2 | 20K TinyStories + 4 books x2 (7.45M tokens) | 5,000 | 2.904 | 2.920 | 2.2 h | First good model |
| R3 | R2 weights, same data | +6,000 | 2.616 | 2.807 | 2.7 h | Better; gains flattening |
| R4a | R3 weights, books only (1.44M tokens), lr 5e-5 | 1,000 | 3.767 | 4.038 | - | No meaningful gain |
| R4b | R3 weights, books only, lr 2e-4 | 600 | 3.772 | 4.066 | - | No meaningful gain |
| R5 | 49.5K TinyStories + 3 Gutenberg books x3 (12.2M tokens) | 8,000 | 2.550 | **2.629** | 4.3 h | Current default model |

## R1 - first attempt (discarded)

- Corpus: 20,000 TinyStories + 4 books repeated 5 times; the last 5% of the
  combined token stream was used for validation.
- **Initial loss was 168.8** instead of about 8.3 (= ln 4096). Cause: PyTorch's
  default N(0, 1) initialisation of `nn.Embedding`, combined with tied
  input/output embeddings, produced enormous initial logits. Training
  recovered, but thousands of steps were wasted.
- **Validation leakage:** because books were repeated 5 times, the "held-out"
  tail of the stream was also present in the training data, so validation loss
  was misleading.
- Fixes: normal initialisation with std 0.02 for linear and embedding layers
  (a regression test now checks that the initial loss is close to ln(vocab));
  validation text is held out before any repetition.

## R2 - baseline after fixes

- Initial loss 8.358 (correct).
- Train 2.904 / best val 2.920 (at iteration 4,750; 3.020 at the last
  evaluation, i.e. evaluation noise of roughly +-0.05 with 20 batches).
- Samples: coherent story structure, dialogue and endings, but broken grammar
  in places (`She was very books`, glued words such as `downthe`).

## R3 - continued training

- Warm-started from R2, 6,000 more iterations, learning rate 3e-4 decayed to
  3e-5, validation averaged over 50 batches.
- Best val 2.807, train 2.616. Validation fell from 2.97 to 2.88 during the
  first 2,500 iterations and to 2.81 by iteration 5,000, then stayed flat.
- Samples: longer and more consistent stories, fewer grammar errors.
  Repetition and character mix-ups remained.

## R4 - fine-tuning on the books (negative result)

- Idea: adapt the R3 model to the style of the four books by fine-tuning only
  on them (1,442,268 training tokens, 75,911 validation tokens).
- lr 5e-5, 1,000 steps: validation 4.14 -> 4.04. lr 2e-4, 600 steps:
  4.10 -> 4.07. Essentially flat.
- Samples became darker in theme but grammar degraded.
- Interpretation: literary prose has far higher loss (about 4.0) than
  TinyStories (about 3.0) for this model, and a 4M-parameter model has too
  little capacity and too little book data (about 1.4M tokens) to learn both
  styles. A text-quality check also showed one book was OCR-garbled (word order
  scrambled); it was removed.

## R5 - current default

- Data: 50,000 TinyStories (49,500 train / 500 validation) plus three clean
  Gutenberg collections (Grimm, Andersen, Aesop for Children), books repeated
  3x in training only.
- 12,208,282 training tokens, 145,083 validation tokens. 8,000 iterations at
  learning rate 6e-4 with cosine decay.
- Validation: 4.37 (iter 250) -> 3.47 (iter 1,000) -> 2.80 (iter 4,000) ->
  2.63 at the end; train 2.55; gap about 0.08 (no overfitting).
- Validation loss was almost flat from iteration 6,500 onwards, indicating the
  model is capacity-limited.

## Sampling observations

Measured by reading samples from the R3/R5 models:

| Setting | Observation |
|---------|-------------|
| temperature 0.2 | Frequent loops ("climbed up the tree and climbed down" repeated) |
| temperature 0.5 - 0.6 | Best balance of coherence and variety |
| temperature 0.8+ | More variety but contradictions and wrong characters |
| repetition penalty 1.05 | Little visible effect |
| repetition penalty 1.1 - 1.15 | Fewer repeats but glued words (`climbit`, `happy!Tim`): penalising recent tokens also penalises the space-prefixed variants, so the model falls back to space-less tokens |
| prompts | Complete sentences with a named character work better than fragments |

## Lessons learned

1. Check that the initial loss is about ln(vocab); anything else signals an
   initialisation or data problem.
2. Hold out validation data **before** repeating or augmenting training data.
3. Inspect training text (`check_books.py`); a single garbled source can hurt.
4. At 4M parameters, more story data helps more than a small amount of
   stylistically different text.
5. Evaluation noise with 20 batches is about 0.05 in loss; use 50 or more.

## Planned experiments

- Train on SimpleStories (`prepare_data.py --dataset simplestories`) and compare samples with R5.
- Train `configs/gpu_14m.json` (about 14M parameters, 256-token context) on a free GPU with more data.
- Instruction-style fine-tuning (`Prompt: ... Story: ...`).
