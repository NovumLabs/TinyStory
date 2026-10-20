"""Tests for the model, initialisation, generation and checkpoints."""

import math

import pytest
import torch

from config import ModelConfig, load_configs
from model import GPT
from train import get_lr, save_checkpoint
from config import TrainConfig


def test_forward_shape_and_loss(tiny_model, tiny_config):
    x = torch.randint(0, tiny_config.vocab_size, (3, 10))
    logits, loss = tiny_model(x, x)
    assert logits.shape == (3, 10, tiny_config.vocab_size)
    assert torch.isfinite(loss)


def test_initial_loss_is_close_to_log_vocab(tiny_model, tiny_config):
    """Regression test: a bad init once produced an initial loss near 168."""
    x = torch.randint(0, tiny_config.vocab_size, (8, 16))
    _, loss = tiny_model(x, x)
    assert abs(loss.item() - math.log(tiny_config.vocab_size)) < 0.3


def test_weights_are_tied(tiny_model):
    assert tiny_model.head.weight is tiny_model.tok_emb.weight


def test_attention_is_causal(tiny_model, tiny_config):
    a = torch.randint(0, tiny_config.vocab_size, (1, 12))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % tiny_config.vocab_size
    la, _ = tiny_model(a)
    lb, _ = tiny_model(b)
    assert torch.allclose(la[:, :-1], lb[:, :-1], atol=1e-5)


def test_sequence_longer_than_block_size_raises(tiny_model, tiny_config):
    x = torch.zeros((1, tiny_config.block_size + 1), dtype=torch.long)
    with pytest.raises(ValueError):
        tiny_model(x)


def test_generate_appends_requested_tokens(tiny_model):
    idx = torch.zeros((1, 3), dtype=torch.long)
    out = tiny_model.generate(idx, max_new_tokens=7, top_k=10)
    assert out.shape == (1, 10)


def test_generate_handles_prompt_longer_than_block_size(tiny_model, tiny_config):
    idx = torch.zeros((1, tiny_config.block_size + 5), dtype=torch.long)
    out = tiny_model.generate(idx, max_new_tokens=2)
    assert out.shape[1] == idx.shape[1] + 2


def test_generate_stops_at_eos(tiny_model):
    idx = torch.zeros((1, 3), dtype=torch.long)
    first = tiny_model.generate(idx, max_new_tokens=1, temperature=0)[0, -1].item()
    out = tiny_model.generate(idx, max_new_tokens=10, temperature=0, eos_id=first)
    assert out.shape[1] == idx.shape[1] + 1


def test_greedy_decoding_is_deterministic(tiny_model):
    idx = torch.zeros((1, 3), dtype=torch.long)
    a = tiny_model.generate(idx, max_new_tokens=8, temperature=0)
    b = tiny_model.generate(idx, max_new_tokens=8, temperature=0)
    assert torch.equal(a, b)


def test_top_k_larger_than_vocab_is_allowed(tiny_model, tiny_config):
    idx = torch.zeros((1, 2), dtype=torch.long)
    out = tiny_model.generate(idx, max_new_tokens=2, top_k=tiny_config.vocab_size * 10)
    assert out.shape == (1, 4)


def test_repetition_penalty_runs(tiny_model):
    idx = torch.zeros((1, 4), dtype=torch.long)
    out = tiny_model.generate(idx, max_new_tokens=5, rep_penalty=1.2, rep_window=4)
    assert out.shape == (1, 9)


def test_generate_restores_training_mode(tiny_config):
    model = GPT(tiny_config).train()
    model.generate(torch.zeros((1, 2), dtype=torch.long), max_new_tokens=1)
    assert model.training


def test_checkpoint_roundtrip(tiny_model, tiny_config, tmp_path):
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, tiny_model, TrainConfig(), iteration=5, val_loss=1.23)
    loaded = GPT.from_checkpoint(path).eval()
    x = torch.randint(0, tiny_config.vocab_size, (2, 8))
    assert torch.allclose(tiny_model(x)[0], loaded(x)[0], atol=1e-6)
    assert loaded.config == tiny_config


def test_legacy_checkpoint_with_pickled_config_loads(tiny_model, tiny_config, tmp_path):
    path = str(tmp_path / "legacy.pt")
    torch.save({"model": tiny_model.state_dict(), "cfg": tiny_config}, path)
    with pytest.warns(UserWarning):
        loaded = GPT.from_checkpoint(path)
    assert loaded.config == tiny_config


def test_invalid_model_config_is_rejected():
    with pytest.raises(ValueError):
        ModelConfig(n_embd=30, n_head=4)
    with pytest.raises(ValueError):
        ModelConfig(dropout=1.5)


def test_lr_schedule_shape():
    cfg = TrainConfig(lr=1.0, min_lr=0.1, warmup_iters=10, max_iters=100)
    assert get_lr(0, cfg) < get_lr(9, cfg) <= 1.0          # warm-up
    assert abs(get_lr(10, cfg) - 1.0) < 1e-9               # peak
    assert get_lr(50, cfg) < get_lr(10, cfg)               # decay
    assert abs(get_lr(100, cfg) - 0.1) < 1e-9              # floor


def test_shipped_configs_are_valid():
    import glob

    paths = glob.glob("configs/*.json")
    assert paths, "expected at least one config in configs/"
    for path in paths:
        model_cfg, train_cfg = load_configs(path)
        assert model_cfg.n_embd % model_cfg.n_head == 0
        assert train_cfg.max_iters > train_cfg.warmup_iters
