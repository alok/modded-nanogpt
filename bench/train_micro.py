from __future__ import annotations

"""Micro training benchmark for NanoGPT variants.

This file implements a **tiny** GPT‐style language model that can be trained
for a few iterations on synthetic data to provide a realistic end-to-end
"tokens / sec" number.  The goal is **not** to reach state-of-the-art
accuracy – only to exercise the forward + backward path including optional
Linear Canonical Transform (LCT) head.

The benchmark purposefully keeps the configuration small so that it runs on a
single H100 GPU inside the 60 minute Modal time-limit.  It is *not* intended
for research-grade results.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim

from torchlayers import LCTLayer

__all__ = ["run"]


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class BenchConfig:
    vocab_size: int = 16_384  # small vocab
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 512
    max_seq_len: int = 256
    batch_size: int = 32

    num_warmup: int = 2  # warm-up steps (to compile kernels)
    num_steps: int = 20  # measured training iterations


# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------


class MiniGPT(nn.Module):
    """Minimal GPT-style model suitable for quick benchmarking."""

    def __init__(self, cfg: BenchConfig, *, use_lct: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_lct = use_lct

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(cfg.max_seq_len, cfg.n_embd))

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.n_embd,
            nhead=cfg.n_head,
            dim_feedforward=4 * cfg.n_embd,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.n_layer)

        if use_lct:
            # Project to complex space, apply LCT along the *embedding* axis,
            # then project back to real before the logits layer.
            self.pre_head = nn.Linear(cfg.n_embd, cfg.n_embd)
            self.lct = LCTLayer(a=0.1, b=1.0, c=0.1, normalized=True)
            self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        else:
            self.pre_head = None
            self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:  # noqa: D401
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> Tensor:  # noqa: D401
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, "Sequence length exceeds the configured maximum."

        x = self.tok_emb(idx) + self.pos_emb[:T]
        x = self.transformer(x)

        if self.use_lct:
            assert self.pre_head is not None  # for mypy
            x = self.pre_head(x)
            x_complex = torch.complex(x, torch.zeros_like(x))
            x = self.lct(x_complex).real

        logits = self.head(x)

        if targets is None:
            return logits

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss


# -----------------------------------------------------------------------------
# Benchmark runner
# -----------------------------------------------------------------------------


def _generate_batch(cfg: BenchConfig, device: torch.device) -> Tuple[Tensor, Tensor]:
    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len), device=device)
    y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len), device=device)
    return x, y


def run(*, use_lct: bool = False) -> Dict[str, float]:
    """Train the mini-model for a few iterations and report throughput metrics."""

    cfg = BenchConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MiniGPT(cfg, use_lct=use_lct).to(device)
    model.train()

    optimiser = optim.AdamW(model.parameters(), lr=3e-4)

    # Dummy data – we reuse the same batch for all iterations to avoid host <-> device traffic.
    data = _generate_batch(cfg, device)

    # Warm-up (compilation, kernel caches, etc.)
    for _ in range(cfg.num_warmup):
        optimiser.zero_grad(set_to_none=True)
        loss = model(*data)
        loss.backward()
        optimiser.step()

    torch.cuda.synchronize()  # ensure warm-up done

    # ------------------------------------------------------------------
    # Timed training loop
    # ------------------------------------------------------------------
    import time as _time

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = _time.perf_counter()
    for _ in range(cfg.num_steps):
        optimiser.zero_grad(set_to_none=True)
        loss = model(*data)
        loss.backward()
        optimiser.step()
    torch.cuda.synchronize()
    total_time_ms = (_time.perf_counter() - t0) * 1000

    total_tokens = cfg.batch_size * cfg.max_seq_len * cfg.num_steps
    tokens_per_sec = total_tokens / (total_time_ms / 1_000)

    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "tokens_per_sec": tokens_per_sec,
        "latency_ms_per_step": total_time_ms / cfg.num_steps,
        "peak_memory_mb": peak_mem_mb,
        "loss_last": float(loss.detach().cpu()),
        "use_lct": use_lct,
    } 