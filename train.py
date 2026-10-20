from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, replace
from typing import Dict, Optional

import numpy as np
import torch

from config import ModelConfig, TrainConfig, load_configs, resolve_device
from model import GPT


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_split(data_dir: str, split: str) -> np.ndarray:
    """Memory-map ``<data_dir>/<split>.bin`` (uint16 token ids)."""
    path = os.path.join(data_dir, f"{split}.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run prepare_data.py first.")
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    device: torch.device,
    rng: np.random.Generator,
):
    """Sample a random batch of (input, target) sequences."""
    if len(data) <= block_size + 1:
        raise ValueError("Dataset is smaller than one block; use more data.")
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


# --------------------------------------------------------------------------- #
# Optimisation helpers
# --------------------------------------------------------------------------- #
def get_lr(it: int, cfg: TrainConfig) -> float:
    """Linear warm-up followed by cosine decay to ``cfg.min_lr``."""
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / cfg.warmup_iters
    if it >= cfg.max_iters:
        return cfg.min_lr
    decay_span = max(1, cfg.max_iters - cfg.warmup_iters)
    ratio = (it - cfg.warmup_iters) / decay_span
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def evaluate(model, data: Dict[str, np.ndarray], cfg: TrainConfig,
             block_size: int, device, rng) -> Dict[str, float]:
    """Average loss over ``cfg.eval_batches`` random batches per split."""
    model.eval()
    out = {}
    for split, d in data.items():
        losses = []
        for _ in range(cfg.eval_batches):
            x, y = get_batch(d, cfg.batch_size, block_size, device, rng)
            losses.append(model(x, y)[1].item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def save_checkpoint(path: str, model: GPT, train_cfg: TrainConfig,
                    iteration: int, val_loss: float) -> None:
    """Atomically write a checkpoint containing only tensors and primitives."""
    payload = {
        "model": model.state_dict(),
        "config": asdict(model.config),
        "train_config": asdict(train_cfg),
        "iter": iteration,
        "val_loss": val_loss,
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the TinyStory model.")
    p.add_argument("--config", help="JSON config file (see configs/).")
    p.add_argument("--data-dir")
    p.add_argument("--out", dest="out_path", help="Checkpoint path (best val loss).")
    p.add_argument("--init-from", help="Warm-start weights from this checkpoint.")
    p.add_argument("--max-iters", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--min-lr", type=float)
    p.add_argument("--warmup-iters", type=int)
    p.add_argument("--eval-every", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--device", help="auto | cpu | cuda")
    p.add_argument("--bf16", action="store_true", default=None,
                   help="Use bfloat16 autocast (CUDA only).")
    p.add_argument("--num-threads", type=int)
    p.add_argument("--log-file", help="Append metrics as JSON lines.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_cfg, train_cfg = load_configs(args.config)
    overrides = {
        k: v for k, v in vars(args).items()
        if k != "config" and v is not None
    }
    train_cfg = replace(train_cfg, **overrides)

    if train_cfg.num_threads:
        torch.set_num_threads(train_cfg.num_threads)
    torch.manual_seed(train_cfg.seed)
    rng = np.random.default_rng(train_cfg.seed)
    device = resolve_device(train_cfg.device)

    # The vocabulary size produced by prepare_data.py is authoritative.
    meta_path = os.path.join(train_cfg.data_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_vocab = json.load(f)["vocab_size"]
        if meta_vocab != model_cfg.vocab_size:
            print(f"Using vocab_size={meta_vocab} from {meta_path} "
                  f"(config said {model_cfg.vocab_size}).")
            model_cfg = replace(model_cfg, vocab_size=meta_vocab)

    data = {
        "train": load_split(train_cfg.data_dir, "train"),
        "val": load_split(train_cfg.data_dir, "val"),
    }

    if train_cfg.init_from:
        model = GPT.from_checkpoint(train_cfg.init_from, device="cpu")
        print(f"Initialised from {train_cfg.init_from}")
    else:
        model = GPT(model_cfg)
    model.to(device)
    model.train()
    block_size = model.config.block_size
    print(f"Parameters: {model.num_parameters() / 1e6:.1f}M | device: {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    use_bf16 = train_cfg.bf16 and device.type == "cuda"
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 else nullcontext()
    )

    best_val = float("inf")
    start = time.time()
    try:
        for it in range(train_cfg.max_iters + 1):
            lr = get_lr(it, train_cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            if it % train_cfg.eval_every == 0 or it == train_cfg.max_iters:
                with autocast:
                    losses = evaluate(model, data, train_cfg, block_size, device, rng)
                elapsed = time.time() - start
                print(f"iter {it} | train {losses['train']:.3f} | "
                      f"val {losses['val']:.3f} | {elapsed:.0f}s", flush=True)
                if train_cfg.log_file:
                    with open(train_cfg.log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"iter": it, "lr": lr, **losses,
                                            "seconds": round(elapsed, 1)}) + "\n")
                if losses["val"] < best_val:
                    best_val = losses["val"]
                    save_checkpoint(train_cfg.out_path, model, train_cfg, it, best_val)
            if it == train_cfg.max_iters:
                break

            x, y = get_batch(data["train"], train_cfg.batch_size, block_size, device, rng)
            with autocast:
                _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    print(f"Training finished. Best val loss: {best_val:.3f} "
          f"(checkpoint: {train_cfg.out_path})")


if __name__ == "__main__":
    main()
