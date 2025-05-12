"""Three-minute micro-benchmark comparing NanoGPT vs NanoGPT + LCT.

This file purposefully keeps the dependency surface tiny so that it can run
inside the Modal stub in < 10 min end-to-end (container + exec).  It uses a
*very* small GPT config to minimise compile time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor

try:
    from train_gpt import GPT  # type: ignore  # Local NanoGPT fork
except ImportError as exc:  # pragma: no cover – benchmark-only path
    raise SystemExit("Could not import GPT modules – ensure NanoGPT code is on PYTHONPATH") from exc

@dataclass
class BenchConfig:
    """Configuration for the benchmark run."""
    vocab_size: int = 1 << 15  # Small vocab for quick testing
    n_layer: int = 4  # Minimal depth for meaningful comparison
    n_head: int = 8
    n_embd: int = 512  # Reduced embedding size for speed
    max_seq_len: int = 256  # Short sequences for quick iteration
    batch_size: int = 32
    num_warmup: int = 5  # Number of warmup iterations
    num_steps: int = 20  # Number of measured iterations

class BenchResult(NamedTuple):
    """Results from a benchmark run."""
    tokens_per_sec: float
    latency_ms: float
    peak_memory_mb: float

def setup_model(use_lct: bool = False) -> GPT:
    """Initialize a tiny GPT model for benchmarking."""
    cfg = BenchConfig()
    model = GPT(
        vocab_size=cfg.vocab_size,
        num_layers=cfg.n_layer,
        num_heads=cfg.n_head,
        model_dim=cfg.n_embd,
        max_seq_len=cfg.max_seq_len,
        use_lct=use_lct
    ).cuda()
    model.eval()  # Benchmark in eval mode
    return model

def generate_dummy_batch(cfg: BenchConfig) -> tuple[Tensor, Tensor]:
    """Create synthetic input data for benchmarking."""
    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len), device="cuda")
    y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len), device="cuda")
    return x, y

def measure_inference(model: GPT, cfg: BenchConfig) -> BenchResult:
    """Run inference benchmark and measure performance metrics."""
    x, y = generate_dummy_batch(cfg)
    
    # Warmup
    for _ in range(cfg.num_warmup):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()
    
    # Measure latency
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    
    for _ in range(cfg.num_steps):
        with torch.no_grad():
            _ = model(x)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    # Calculate metrics
    total_time = end_time - start_time
    total_tokens = cfg.batch_size * cfg.max_seq_len * cfg.num_steps
    tokens_per_sec = total_tokens / total_time
    latency_ms = (total_time / cfg.num_steps) * 1000
    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # Convert to MB
    
    return BenchResult(tokens_per_sec, latency_ms, peak_memory)

def run(use_lct: bool = False) -> dict[str, float]:
    """Run the benchmark and return performance metrics."""
    cfg = BenchConfig()
    model = setup_model(use_lct)
    
    # Run benchmark
    result = measure_inference(model, cfg)
    
    # Return metrics as a dictionary
    return {
        "tokens_per_sec": result.tokens_per_sec,
        "latency_ms": result.latency_ms,
        "peak_memory_mb": result.peak_memory_mb,
        "batch_size": cfg.batch_size,
        "seq_len": cfg.max_seq_len,
        "model_size": sum(p.numel() for p in model.parameters()) / 1e6,  # Millions of parameters
    }
