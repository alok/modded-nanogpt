# AGENT PLAN – NeurIPS 2025 LCT-NanoGPT Project

_Last updated: 2025-05-11 23:42 EDT_
_Last updated: 2025-05-11 22:59 EDT_

## 0. TL;DR

Write a polished NeurIPS 2025 extended abstract showcasing a Linear Canonical Transform (LCT) layer inside NanoGPT; ship reproducible code, benchmarks, and documentation.

---

## 1. Immediate Next Actions

| Priority | Task                                            | Linked § | When  |
| -------- | ----------------------------------------------- | -------- | ----- |
| 🔥        | Build repo skeleton & smoke‐test MVP `LCTLayer` | 3.1/3.2  | Today |
| 🔥        | Draft abstract outline (`paper/outline.md`)     | 4        | Today |
| 🔥        | Wire into NanoGPT `--use-lct` (concat+proj)     | 3.7      | Today |
| 🆕        | Oracle tests (Fourier, Laplace)                 | 3.3      | Today |
| 🆕        | Quick benchmark script `just bench:lct`         | 3.8      | Today |
| ⚠️        | Update results table & abstract numbers         | 4        | Today |
| ℹ️        | Tag preview `v0.0.1`                            | 3.10     | Today |

---

## 2. Milestones

1. **LCT Minimum Viable Layer**  
   Code complete & passes unit tests.
2. **Integration**  
   Swap NanoGPT's `nn.Linear` with `LCTLayer`; training script runs.
3. **Benchmark**  
   Tokens/sec baseline vs LCT plotted; results in `img/`.
4. **Special-Case Compliance**  
   All special-case tests (unitarity, Laplace, fractional Fourier, Fresnel) pass.
5. **Paper**  
   Extended abstract PDF passes NeurIPS style checker.
6. **Submission Package**  
   Tagged release `v0.1.0` on GitHub with Zenodo DOI.

---

## 3. Detailed Implementation Plan (synthesised from `paper/main.tex` & `FLCTISIEONE.tex`)

### 3.0 Scope

Deliver a production-grade, differentiable Linear Canonical Transform (LCT) layer for PyTorch and integrate it into NanoGPT.  We adopt the fast chirp–FFT–chirp decomposition presented in §2 of _FLCTISIEONE.tex_, achieving **O(N log N)** complexity while leveraging cuFFT on GPU.

### 3.1 Repository Skeleton

```
torchlayers/
  __init__.py
  lct.py                # LCTLayer implementation
  functional/
    __init__.py
    lct.py              # low-level helpers (chirp factors, kernels)
tests/
  test_lct.py           # core: FFT reduction & inverse accuracy
  test_lct_special.py   # Laplace, Fresnel, FrFT, unitarity checks
bench/
  bench_lct.py          # micro-benchmark & profile script
justfile                # `test:all`, `lint`, `bench:lct`, …
docs/
  lct_math.md           # derivations, parameterisation
```

### 3.2 Core Algorithm (LCTLayer)

**Implementation order guideline:** Begin by shipping a **fully-general, CPU-only** version of `LCTLayer` with _no_ special-case branches or performance tweaks.  Ensure the layer works standalone and passes the validation suite **before** attempting GPU acceleration, NanoGPT integration, or any optimisation work noted later in this plan.

1. **Parameterisation**  
   • Three learnable **complex64** scalars `a, b, c` (`nn.Parameter`, stored as real–imag pairs ⇒ 6 real DOF).  
   • Compute `d` on-the-fly such that `ad − bc = 1`.  
   • Regularise near-singular branch (`|a|≤ε`) with Taylor fallback.
2. **Forward pass** (`b ≠ 0`)  
   Chirp–FFT–chirp:  
   `y = C₀ · exp(iπ a/b · x²) · FFT[ exp(iπ/b · x²) · x ]`,  where `C₀ = 1/√|2π b|`.
3. **Inverse pass**  
   Use symplectic inverse parameters `(d, −b, −c, a)` and reuse the same kernel.
4. **Batch & Autograd**  
   • Sequence batching via broadcasting / `torch.vmap`.  
   • Cache chirp tensors with `register_buffer`; rebuild when parameters update.
5. **Mixed precision**  
   All constants cast to `x.dtype`; supports bf16 & (capability-gated) fp8.
6. **(Experimental) Lie-algebra parameterisation – SL(2,ℂ)**  
   • Model the generator `M' = p₁ H + p₂ X + p₃ Y ∈ 𝔰𝔩(2,ℂ)` with traceless basis `H = [[1,0],[0,-1]]`, `X = [[0,1],[0,0]]`, `Y = [[0,0],[1,0]]`.  
   • Learn complex coefficients `pᵢ ∈ ℂ` (6-real DOF) and obtain the canonical matrix via `M = torch.matrix_exp(M') ∈ SL(2,ℂ)` which always satisfies `det M = 1`.  
   • Parse `(a,b,c,d)` from `M` and route through the same chirp–FFT–chirp kernel.  
   • Unlocks non-unitary special cases (e.g.
     Laplace) and removes division–by–zero pitfalls when solving for `d`.  
   • Gate behind `lie_param=True`; **deferred** until MVP is merged and stabilised.
7. **Degeneracy safeguard**  
   • If `|b| < ε` (default `ε = 1e-4`) automatically switch to the `b ≈ 0` path with fast approximate resampling (identity shortcut when `d ≈ 1`).  
   • Emit a `logger.debug` or `wandb` counter (`lct/degen_hits`) each time the branch fires to monitor training stability.

### 3.3 Validation Suite

• FFT reduction: `(a,b,c) = (0,1,0)` → L2 < 1e-6 vs `torch.fft.fft`  
• Inverse consistency: `x ≈ layer.inverse(layer(x))` (max |err| < 1e-6)  
• Laplace, Fresnel, fractional Fourier (α ∈ {π/4, π/2, 3π/4})  
• Unitarity: `LCT · LCTᴴ ≈ I` for random N ≤ 256.

### 3.4 API & Docs

```python
layer = LCTLayer(size=1024, init=(0., 1., 0.), learnable=True)
y = layer(x)           # x: (batch, size)
z = layer.inverse(y)   # z ≈ x
```

Mathematical exposition lives in `docs/lct_math.md`; docstrings follow NumPy style and are rendered by Sphinx.

### 3.5 Performance Engineering

• Prefer `torch.fft.rfft` for real inputs.  
• Cache cuFFT plans across calls.  
• Provide `layer.cuda_graph()` for capture.  
• FP8 path gated by `torch.cuda.get_device_capability() ≥ (9,0)`.

### 3.6 Rule Compliance (`.cursor/rules/neurips_2025_plan.mdc`)

* `uv` for deps, `ruff/black/mypy` enforced in CI.  
* One git commit per atomic change; update `CHANGELOG.md` after public API edits.

### 3.7 NanoGPT Integration

• Add CLI flag `--use-lct` to `train_gpt*.py`.  
• Insert `LCTLayer` **along the sequence length dimension L**, immediately _before_ self-attention (`X → LCT → QKV`).  
• Represent complex output as `[Re ; Im]` concatenation ⇒ doubles the channel dimension fed to attention; an initial linear bottleneck can project back when required.  
• **Immediately** apply `nn.Linear(2d → d)` to restore the original width and contain FLOPs; weights initialised to block‐wise identity.  
• Alternate strategies (`magnitude`, `real-only`) kept for ablation (§3.11).  
• Account for RoPE interactions – monitor if LCT learns to normalise or warp positional phases.  
• Keep param-count parity (`in_features == out_features`) when concatenation is folded into the subsequent linear.  
• Forward-only sanity test `tests/test_nano_integration.py`.

### 3.8 Benchmark Harness

• `just bench:lct` logs tokens/s & VRAM to `records/YYYYMMDD_LCTBench/`.  
• Optional wandb upload when key present.

### 3.9 Continuous Integration

GitHub Action `{ubuntu-latest, macos-13}` × `{3.12, nightly}` running:
`uv pip install -e .[dev] && ruff . && black --check . && mypy --strict . && pytest -q`.

### 3.10 Packaging & Release

• Export `LCTLayer` in `torchlayers/__init__.py`.  
• Version `0.0.1` → TestPyPI; bump to `0.1.0` for camera-ready.  
• Create Zenodo DOI tag `v0.1.0`.

### 3.11 Ablation & Analysis

* **Layer impact** – baseline vs `--use-lct`.
* **Placement** – pre-attention vs post-attention vs dedicated block. _(P0)_
* **Complex-to-real mapping** – `[Re ; Im]` concat vs magnitude vs real-only. _(P0)_
* **Fixed vs learnable** – freeze parameters to FFT / FrFT / Fresnel vs fully learnable. _(P1)_
* **RoPE interaction** – compare runs with/without rotary embeddings. _(P1)_
* **Degeneracy threshold** – sweep `ε` for the `b≈0` switch. _(P2)_
Benchmarks report: tokens/s, wall-clock to target loss, FLOPs step overhead, memory, and parameter histograms `(a,b,c,d)`.

---

## 4. Backlog / Ideas

* Investigate constraining `(a,b,c)` on the tangent space of `Sp(2,ℝ)` to improve conditioning.
* Explore automatic parameter initialisation from data-driven moment matching.
* FP8 path: benchmark NF4 vs FP8-E4M3.
* Use `torch.compile` (TorchDynamo) to fuse chirp multiplications.
* Ensure conjugate 2π convention (`normalized=True`) matches NumPy ortho mode.

---

## 5. Decisions & Rationale (Chronological)

| Date       | Decision                                | Why                                 |
| ---------- | --------------------------------------- | ----------------------------------- |
| 2025-05-11 | Use Cursor Rule `neurips_2025_plan.mdc` | Always remind AI of deliverables    |
| 2025-05-11 | Deprecate `.cursorrules`                | Proper location is `.cursor/rules/` |

---

## 6. Glossary

* **LCT** – Linear Canonical Transform, param \(a,b,c\) governing affine symplectic mapping. _(≠ "linear chirp transform")_
* **FFT** – Fast Fourier Transform; recovered when \(a=0, b=1, c=0\).

---

## 7. Next Update Trigger

When any task completes, open this file, tick box ✅, append to _Decisions_ if applicable, and bump timestamp at top.

---

## 8. Linear Canonical Transform (LCT) Implementation Guide

### Overview

This section distills and refines the LCT implementation plan into a clear, executable guide for any O3 instance (or developer) to follow.  It focuses on:

* Practical module structure & responsibilities
* Parameter handling across general & special‐case regimes
* Caching/pre-computation strategy for efficiency
* Exhaustive test checkpoints and expected tolerances

The content is fully aligned with the `modded-nanogpt` environment, project conventions, and deliverables enumerated in §3.

---

### 8.1  Parameterisation & Key Equations

The continuous‐time Linear Canonical Transform is defined by
\[
    \begin{pmatrix}a & b \\ c & d\end{pmatrix} \in \mathrm{SL}(2,\mathbb R),\qquad ad-bc=1.
\]

1. **Generic case** \(b \neq 0\)
\[
X(u)=\frac{1}{\sqrt{i\,b}}\,e^{ i\pi\frac{d}{b}u^2}\int_{-\infty}^{\infty} e^{-i2\pi\frac{1}{b}ut}\; e^{ i\pi\frac{a}{b}t^2}\,x(t)\,dt.
\]

2. **Degenerate case** \(b = 0\)
\[
X(u)=\sqrt{d}\;e^{ i\pi c d u^2}\,x(d u).
\]

Special instances: identity, Fourier \((0,1,-1,0)\), fractional Fourier \((\cos\theta, \sin\theta, -\sin\theta, \cos\theta)\), and pure scaling.

---

### 8.2  Discrete-Time Strategy (Chirp–FFT–Chirp)

* **Input chirp**: \(C_{\text{in}}[n]=e^{ i\pi\frac{a}{b}n^2 }\)
* **FFT step**: one call to `torch.fft.fft` along the transform axis (optionally `fftshift`).
* **Output chirp**: \(C_{\text{out}}[m]=e^{ i\pi\frac{d}{b}m^2 }\).
* **Normalisation**: multiply by \(\tfrac{1}{\sqrt{i b N}}\) so the operator is unitary in discrete \(\ell^2\).

When \(b=0\), bypass the FFT and instead resample + chirp multiply, using `torch.nn.functional.grid_sample` for non-integer scaling factors.

---

### 8.3  Functional Decomposition

```python
# torchlayers/functional/lct.py
from __future__ import annotations

import math
from typing import Final

import torch

Complex = torch.complex64  # local alias for brevity

π: Final[float] = math.pi


def _chirp_phase(length: int, coeff: float, /, centered: bool = True, *, device, dtype) -> torch.Tensor:
    """Return exp( i π * coeff * n^2 ) as 1-D complex tensor."""
    n = torch.arange(length, device=device, dtype=dtype)
    if centered:
        n = n - length // 2
    phase = 1j * π * coeff * n**2
    return torch.exp(phase)


def linear_canonical_transform(
    x: torch.Tensor,
    *,
    a: float,
    b: float,
    c: float,
    d: float,
    dim: int = -1,
    centered: bool = True,
) -> torch.Tensor:
    """Apply the discrete LCT along *dim*.

    Pre-conditions: ``abs(a*d - b*c - 1) < 1e-6``.
    Returns a *complex* tensor regardless of the real/complex nature of *x*.
    """

    if abs(a * d - b * c - 1) > 1e-6:
        raise ValueError("LCT parameters must satisfy ad - bc = 1.")

    x = x.to(torch.complex64)
    N = x.size(dim)

    if b != 0:
        # --- generic path ---
        coeff_in = a / b
        coeff_out = d / b

        chirp_in = _chirp_phase(N, coeff_in, centered=centered, device=x.device, dtype=x.dtype)
        chirp_out = _chirp_phase(N, coeff_out, centered=centered, device=x.device, dtype=x.dtype)

        # broadcast multiply
        x = x * torch.moveaxis(chirp_in, 0, dim)
        X = torch.fft.fft(x, dim=dim) / math.sqrt(N)
        X = X * torch.moveaxis(chirp_out, 0, dim)

        const = 1 / torch.sqrt(1j * torch.tensor(b, dtype=x.dtype, device=x.device))
        return const * X

    # --- b == 0 path ---
    scale = d
    sqrt_d = torch.sqrt(torch.tensor(d, dtype=x.dtype, device=x.device))

    length = torch.arange(N, device=x.device, dtype=torch.float32)
    if centered:
        length = length - N // 2
    t = scale * length  # sampled positions

    # normalise grid to [-1,1] for grid_sample
    grid = (2 * t / max(N - 1, 1)).unsqueeze(0).unsqueeze(-1)  # shape (1,N,1)
    x_unsq = x.unsqueeze(1)  # add channel dim for grid_sample: (B,1,N)
    resampled = torch.nn.functional.grid_sample(
        x_unsq.real, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    ).squeeze(1) + 1j * torch.nn.functional.grid_sample(
        x_unsq.imag, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    ).squeeze(1)

    chirp = _chirp_phase(N, c * d, centered=centered, device=x.device, dtype=x.dtype)
    resampled = sqrt_d * resampled * torch.moveaxis(chirp, 0, dim)
    return resampled
```

The high-level `LinearCanonicalTransform` module simply wraps the above, pre-computing chirps as `register_buffer`s when `seq_len` is known.

---

### 8.4  Caching & Mixed Precision

* **Buffers**: `chirp_in`, `chirp_out`, and the normalisation scalar are registered buffers ⇒ auto-moved with `.to(device)`.
* **Precision**: construct chirps in `float64` then cast to `x.dtype` for minimal phase error.
* **cuFFT plans**: PyTorch internally caches; no explicit management needed.

---

### 8.5  Verification Checklist (pytest)

1. _Fourier oracle_ – `(0,1,-1,0)` matches `torch.fft.fft` (tol ≤ 1e-6).
2. _Identity_ – `(1,0,0,1)` returns input exactly.
3. _Inverse_ consistency – `LCT(M⁻¹)(LCT(M)(x)) ≈ x`.
4. _Unitarity_ – `‖LCT(x)‖₂ == ‖x‖₂` within 1e-6.
5. _Group property_ – FrFT(θ₁) ∘ FrFT(θ₂) ≈ FrFT(θ₁+θ₂).
6. _Batch broadcasting_ – compare loop vs batched run.

All tests live in `tests/test_lct.py` & `tests/test_lct_special.py`.

---

### 8.6  Integration Notes

* CLI flag `--use-lct` toggles replacement of `nn.Linear` with `LCTLayer`.
* Preserve parameter count parity to ensure checkpoint compatibility.
* Expect complex tensors downstream ⇒ if real activations required, split channels or take magnitude/phase.

---

### 8.7  Future Extensions

* **Learnable parameters** constrained via angle/rapidity re-parameterisation.
* **2-D LCT** for vision tasks.
* **Chirp-Z** implementation for exact non-integer 1/\(b\) sampling.

---

_This guide supersedes earlier terse notes in §3 where overlap exists._

### 1.1  Execution Sprint Checklist (one-day)
1. `pytest -q` → all FFT/Laplace tests green.
2. Wire `--use-lct` flag & `[Re;Im]` concat+bottleneck in `train_gpt*.py`.
3. Run `just bench:lct` → record tokens/s vs baseline.
4. Draft `paper/outline.md` with 5-part structure (motivation, method, expt, results, impact).
5. Tick items in §1 table, commit & tag `v0.0.1-preview`.
