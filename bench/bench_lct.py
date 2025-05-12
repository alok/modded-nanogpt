"""Three-minute micro-benchmark comparing NanoGPT vs NanoGPT + LCT.

This file purposefully keeps the dependency surface tiny so that it can run
inside the Modal stub in < 10 min end-to-end (container + exec).  It uses a
*very* small GPT config to minimise compile time.
"""

from __future__ import annotations

# flake8: noqa: F401
# pyright: reportMissingImports=false, reportGeneralTypeIssues=false

import time
from argparse import ArgumentParser

import torch

try:
    from train_gpt import GPT, GPTConfig  # type: ignore  # Local NanoGPT fork
except ImportError as exc:  # pragma: no cover – benchmark-only path
    raise SystemExit("Could not import GPT modules – ensure NanoGPT code is on PYTHONPATH") from exc

from torchlayers.lct import LCTLayer

__all__ = ["run"]


def _build_model(use_lct: bool) -> GPT:  # type: ignore[name-defined]
    """Return a tiny GPT model optionally patched with an LCT output head."""

    cfg = GPTConfig(vocab_size=1 << 15, n_layer=4, n_head=8, n_embd=512)  # type: ignore[arg-type]
    model = GPT(cfg).cuda()  # type: ignore[arg-type]

    if use_lct:
        # Replace the projection layer with a learnable LCT.  This assumes the
        # stock GPT exposes ``model.proj`` – adapt if the API is different.
        if hasattr(model, "proj"):
            model.proj = LCTLayer().cuda()
        else:  # pragma: no cover
            raise AttributeError("GPT model does not expose a 'proj' attribute to patch.")

    return model


def run(use_lct: bool = False) -> float:
    """Return throughput in tokens/sec measured over a 300-second window."""

    model = _build_model(use_lct)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batch = torch.randint(0, 1 << 15, (32, 128), device="cuda")

    start = time.perf_counter()
    processed = 0
    while (elapsed := time.perf_counter() - start) < 300:  # 5 minutes safety
        loss = model(batch)  # type: ignore[call-arg]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        processed += batch.numel()

    return processed / elapsed


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--lct", action="store_true", help="Enable LCT layer patch")
    args = ap.parse_args()
    tok_per_sec = run(args.lct)
    tag = "LCT" if args.lct else "baseline"
    print(f"{tag}: {tok_per_sec:,.1f} tok/s")
