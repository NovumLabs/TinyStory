from __future__ import annotations

import argparse
from typing import Optional

import torch
from tokenizers import Tokenizer

from config import EOT_TOKEN, resolve_device
from model import GPT


def generate_text(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.6,
    top_k: Optional[int] = 40,
    rep_penalty: float = 1.0,
    rep_window: int = 24,
    seed: Optional[int] = None,
) -> str:
    """Continue ``prompt`` and return the decoded text.

    Generation stops early when the model emits the end-of-text token, which
    marks the end of a story. An empty prompt starts from that token.
    """
    device = next(model.parameters()).device
    eos_id = tokenizer.token_to_id(EOT_TOKEN)
    ids = tokenizer.encode(prompt).ids if prompt else []
    if not ids:
        ids = [eos_id if eos_id is not None else 0]
    if seed is not None:
        torch.manual_seed(seed)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=eos_id,
        rep_penalty=rep_penalty,
        rep_window=rep_window,
    )
    return tokenizer.decode(out[0].tolist())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text with a trained model.")
    p.add_argument("prompt", nargs="?", default="Once upon a time",
                   help="Text the story should start with.")
    p.add_argument("--ckpt", default="ckpt.pt", help="Checkpoint path.")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--temperature", type=float, default=0.5,
                   help="Lower is safer but more repetitive (0 = greedy).")
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--rep-penalty", type=float, default=1.0,
                   help="Values slightly above 1.0 (e.g. 1.05) reduce repetition.")
    p.add_argument("--rep-window", type=int, default=24)
    p.add_argument("--max-new-tokens", type=int, default=180)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model = GPT.from_checkpoint(args.ckpt, device=device)
    tokenizer = Tokenizer.from_file(args.tokenizer)

    for i in range(args.num_samples):
        seed = None if args.seed is None else args.seed + i
        text = generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            rep_penalty=args.rep_penalty,
            rep_window=args.rep_window,
            seed=seed,
        )
        if args.num_samples > 1:
            print(f"--- sample {i + 1} ---")
        print(text)
        if args.num_samples > 1:
            print()


if __name__ == "__main__":
    main()
