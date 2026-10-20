from __future__ import annotations

import warnings
from dataclasses import asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig

# Backward-compatible alias: checkpoints written by early versions of this
# project pickled the config as ``model.Config``.
Config = ModelConfig


class Attention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, c: ModelConfig):
        super().__init__()
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)
        self.proj = nn.Linear(c.n_embd, c.n_embd)
        self.n_head = c.n_head
        self.drop = c.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    """Pre-LayerNorm transformer block: attention followed by an MLP."""

    def __init__(self, c: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.attn = Attention(c)
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(c.n_embd, 4 * c.n_embd),
            nn.GELU(),
            nn.Linear(4 * c.n_embd, c.n_embd),
            nn.Dropout(c.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Decoder-only language model with tied input/output embeddings."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Small-normal initialisation.

        This matters: with tied embeddings, PyTorch's default ``N(0, 1)``
        embedding init produces huge initial logits (loss in the hundreds).
        Using a standard deviation of 0.02 starts training near ``ln(vocab)``.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        """Number of unique trainable parameters (tied weights counted once)."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block_size {self.config.block_size}"
            )
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.ln_f(self.blocks(x))
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @staticmethod
    def _penalize_repeats(
        logits: torch.Tensor, idx: torch.Tensor, penalty: float, window: int
    ) -> torch.Tensor:
        """Down-weight tokens that appeared in the last ``window`` positions."""
        logits = logits.clone()
        for b in range(idx.size(0)):
            recent = idx[b, -window:].unique()
            vals = logits[b, recent]
            logits[b, recent] = torch.where(vals > 0, vals / penalty, vals * penalty)
        return logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        eos_id: Optional[int] = None,
        rep_penalty: float = 1.0,
        rep_window: int = 24,
    ) -> torch.Tensor:
        """Autoregressively extend ``idx`` (shape ``(B, T)``).

        Args:
            idx: Prompt token ids.
            max_new_tokens: Maximum number of tokens to append.
            temperature: Softmax temperature. ``<= 0`` selects greedy decoding.
            top_k: Sample only from the ``k`` most likely tokens (``None``/0 disables).
            eos_id: Stop once every sequence in the batch has produced this token.
            rep_penalty: Values above 1.0 discourage recently used tokens.
            rep_window: Number of recent tokens considered by the penalty.
        """
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
            if rep_penalty != 1.0:
                logits = self._penalize_repeats(logits, idx, rep_penalty, rep_window)
            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                if top_k:
                    k = min(top_k, logits.size(-1))
                    v, _ = torch.topk(logits, k)
                    logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
        if was_training:
            self.train()
        return idx

    @classmethod
    def from_checkpoint(cls, path: str, device="cpu") -> "GPT":
        """Load a model from a checkpoint written by ``train.py``.

        Current checkpoints store the config as a plain dict and are loaded
        with ``weights_only=True``. Legacy checkpoints (config pickled as an
        object) fall back to full unpickling, which can execute arbitrary code:
        only load legacy files you created yourself.
        """
        try:
            ckpt = torch.load(path, map_location=device, weights_only=True)
        except Exception:
            warnings.warn(
                "Falling back to unsafe unpickling for a legacy checkpoint.",
                stacklevel=2,
            )
            ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", ckpt.get("cfg"))
        if cfg is None:
            raise KeyError("Checkpoint contains neither 'config' nor 'cfg'.")
        if not isinstance(cfg, dict):
            cfg = asdict(cfg)
        model = cls(ModelConfig(**cfg))
        model.load_state_dict(ckpt["model"])
        return model.to(device)
