# AGENT PLAN – NeurIPS 2025 LCT-NanoGPT Project

_Last updated: 2025-05-11 23:42 EDT_

## 0. TL;DR
Write a polished NeurIPS 2025 extended abstract showcasing a Linear Canonical Transform (LCT) layer inside NanoGPT; ship reproducible code, benchmarks, and documentation.

---

## 1. Immediate Next Actions

| Priority | Task                                                                                     | Linked § | ETA |
|----------|-------------------------------------------------------------------------------------------|----------|-----|
| 🔥       | Build repository skeleton (files & stubs in §3.1)                                         | 3.1      | 05-12 |
| 🔥       | Draft abstract outline (`paper/outline.md`)                                               | 4        | 05-12 |
| 🔥       | Implement differentiable `LCTLayer.forward` & `inverse` (+ smoke test)                     | 3.2      | 05-13 |
| 🔥       | Integrate `LCTLayer` into NanoGPT via `--use-lct` flag                                     | 3.7      | 05-13 |
| 🆕       | Add oracle tests for Laplace / Fresnel / FrFT cases                                        | 3.3      | 05-14 |
| 🆕       | CI workflow `ci/python-tests.yml` (ruff + mypy + pytest)                                   | 3.9      | 05-14 |
| 🆕       | Benchmark script `just bench:lct` + wandb logging                                          | 3.8      | 05-15 |
| ⚠️       | Update results table in `paper/main.tex` after first benchmark run                         | 4        | 05-16 |
| ℹ️       | Preview release `v0.0.1` to TestPyPI                                                       | 3.10     | 05-17 |

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
Deliver a production-grade, differentiable Linear Canonical Transform (LCT) layer for PyTorch and integrate it into NanoGPT.  We adopt the fast chirp–FFT–chirp decomposition presented in §2 of *FLCTISIEONE.tex*, achieving **O(N log N)** complexity while leveraging cuFFT on GPU.

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
1. **Parameterisation**  
   • Three learnable scalars `a, b, c` (`nn.Parameter`).  
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
• Replace projection layer with `LCTLayer` when flag is set.  
• Maintain param-count parity (`in_features == out_features`).  
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

* **LCT** – Linear Canonical Transform, param \(a,b,c\) governing affine symplectic mapping.
* **FFT** – Fast Fourier Transform; recovered when \(a=0, b=1, c=0\).

---

## 7. Next Update Trigger
When any task completes, open this file, tick box ✅, append to *Decisions* if applicable, and bump timestamp at top.
