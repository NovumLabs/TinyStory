from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple

#: Special token that separates documents in the training corpus.
EOT_TOKEN = "<|endoftext|>"


@dataclass
class ModelConfig:
    """Hyperparameters of the GPT-style decoder-only transformer."""

    vocab_size: int = 4096
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")


@dataclass
class TrainConfig:
    """Training-loop settings."""

    batch_size: int = 32
    max_iters: int = 8000
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_iters: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 50
    seed: int = 1337
    data_dir: str = "data"
    out_path: str = "ckpt.pt"
    init_from: Optional[str] = None  # warm-start weights from an existing checkpoint
    device: str = "auto"  # "auto", "cpu" or "cuda"
    bf16: bool = False  # bfloat16 autocast (CUDA only)
    num_threads: Optional[int] = None  # torch CPU threads (None = PyTorch default)
    log_file: Optional[str] = None  # optional JSON-lines metrics log


def _build(cls: type, values: Dict[str, Any], section: str):
    """Instantiate ``cls`` from ``values``, rejecting unknown keys."""
    valid = {f.name for f in fields(cls)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(
            f"Unknown key(s) in '{section}' section: {sorted(unknown)}. "
            f"Valid keys: {sorted(valid)}"
        )
    return cls(**values)


def load_configs(path: Optional[str] = None) -> Tuple[ModelConfig, TrainConfig]:
    """Load ``(ModelConfig, TrainConfig)`` from a JSON file.

    The file must contain optional ``"model"`` and ``"train"`` objects. Missing
    keys fall back to the dataclass defaults. With ``path=None`` the defaults
    are returned.
    """
    if path is None:
        return ModelConfig(), TrainConfig()
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    unknown = set(raw) - {"model", "train"}
    if unknown:
        raise ValueError(f"Unknown top-level key(s) in {path}: {sorted(unknown)}")
    model_cfg = _build(ModelConfig, raw.get("model", {}), "model")
    train_cfg = _build(TrainConfig, raw.get("train", {}), "train")
    return model_cfg, train_cfg


def resolve_device(name: str = "auto"):
    """Return a ``torch.device`` for ``name`` ("auto", "cpu", "cuda", ...)."""
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
