"""Micro-benchmark for the (forthcoming) LCTLayer.

Usage::

    $ python -m bench.bench_lct --size 1024 --device cuda

At the moment this is a stub that prints a *not yet implemented* warning so
that the CLI wiring can be tested before the kernel is ready.
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext

import torch

# Ensure the import works even before LCT implementation is complete.
from torchlayers.lct import LCTLayer  # noqa: E402


def _parse_args() -> argparse.Namespace:  # noqa: D401
    p = argparse.ArgumentParser(description="Benchmark the LCT layer")
    p.add_argument("--size", type=int, default=1024, help="transform size (N)")
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    p.add_argument("--repeat", type=int, default=50, help="# iterations")
    return p.parse_args()


def main() -> None:  # noqa: D401
    args = _parse_args()

    dev_ctx = torch.device(args.device)

    x = torch.randn(args.size, device=dev_ctx, dtype=torch.complex64)
    layer = LCTLayer().to(dev_ctx)  # default params (behave like FFT)

    # Warm-up (esp. important for CUDA)
    try:
        layer(x)
    except NotImplementedError:
        print("[WARN] LCT kernel not implemented yet – exiting benchmark early.")
        return

    torch.cuda.synchronize() if dev_ctx.type == "cuda" else None

    start = time.perf_counter()
    for _ in range(args.repeat):
        _ = layer(x)
    torch.cuda.synchronize() if dev_ctx.type == "cuda" else None
    stop = time.perf_counter()

    print(f"elapsed: {(stop-start)/args.repeat*1e3:.3f} ms per call")


if __name__ == "__main__":
    main()
