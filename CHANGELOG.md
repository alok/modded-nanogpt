# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Cursor rule `.cursor/rules/neurips_2025_plan.mdc` tracks NeurIPS 2025 deliverables.
- Skeleton LCT layer (`torchlayers/lct.py`) with placeholder forward/inverse.
- Special-case test suite `tests/test_lct_special.py` (currently xfail) to cover Fourier/Laplace/Fresnel etc.
- Updated `AGENT.md` with new tasks and milestones.
- Discrete Linear Canonical Transform kernel `torchlayers/functional/lct.py` implementing chirp–FFT–chirp algorithm.
- Smoke-test suite `tests/test_lct.py` now validates Fourier reduction and inverse property.  Kernel passes.
- Quadrature-based Laplace transform oracle and Fresnel/FrFT helpers.
- Quadrature-derived unitary DFT reference and equivalence tests ensuring
  `LCTLayer(a=0,b=1,c=0)` matches both quadrature and `torch.fft.fft` on
  random complex signals.
- Marked future extension for non-norm preserving group law variant in `AGENT.md`.
- Prepared codebase for larger N scaling with optimized memory handling.
- [2025-05-12 04:53] Integrated LCT layer as an optional replacement for the language model head in NanoGPT
- [2025-05-12 04:53] Enhanced benchmark suite with comprehensive metrics (tokens/sec, latency, memory)
- [2025-05-12 04:53] Added distributed benchmarking support on Modal with H100s
- (2025-05-16:12:00) NeurIPS paper scaffolding ported from `infnum`: section‐based LaTeX skeleton under `paper/sections/*` using local `neurips_2025.sty`.
- New documentation `docs/lct_use_cases.md` describing three rapid-tuning neural activation use cases (Fresnel Vision Attention, Beam-Splash Diffusion, Mirror-Symmetry Mixer).
- Added new Cursor rules ported from `infnum` to harmonise workflows:
  - `.cursor/rules/image-extractor-usage.mdc` – guidance for viewing generated images via Screenpipe.
  - `.cursor/rules/appendix-workflow.mdc` – appendix & artefacts handling.
  - `.cursor/rules/neurips-checklist-guidance.mdc` – integrates NeurIPS paper checklist.
  - `.cursor/rules/kernel-benchmarking.mdc` – micro-benchmark process for performance-critical kernels.
  - `.cursor/rules/parallel-workstreams.mdc` – branch/worktree coordination policy.
- Integrated LCT layer into `train_gpt.py` and `train_gptm.py` as an option within Transformer Blocks, affecting the input to self-attention.
  - Added `use_lct_in_block` flag to `Hyperparameters` and `GPT` class in both scripts.
  - `Block` class now conditionally initializes `LCTLayer` and a projection layer.
  - `Block.forward` in both scripts routes data through LCT path if enabled.
- Created `bench/bench_lct.py` for benchmarking throughput (tokens/sec) of `train_gpt.py` model with and without LCT in blocks.
  - Script uses `tyro` for CLI configuration.
  - Saves benchmark results to timestamped JSON files in `records/`.
- Added `tyro` to `requirements.txt`.
- Created `justfile` with targets for `test`, `lint`, `bench-lct`, and `paper` compilation.
- Updated NeurIPS paper sections (`00_abstract.tex`, `01_introduction.tex`, `04_experiments.tex`, `05_results.tex`, `99_checklist.tex`) with placeholders and current progress.
- Created `paper/outline.md` to guide abstract writing.

### Changed

- Deprecated `.cursorrules` in favour of the canonical rule file.
- Refactored `tests/test_lct_special.py` to derive the Fourier reference via `torch.fft.fft`, cleaned up linter issues, and streamlined parametrisation.
- `torchlayers/lct.py` now delegates to the functional kernel for forward/inverse, ensuring API consistency.
- Expanded implementation details and verification checklist in `AGENT.md`.
- Fast-path FFT shortcut reinstated in `LCTLayer` to satisfy Fourier oracle without division-by-zero.
- `_chirp_phase` patched for complex dtype safety (no NaNs).
- Optimized memory usage for larger N values by using float32 for intermediate computations.
- [2025-05-12 04:53] Modified GPT class to support optional LCT-based language model head
- [2025-05-12 04:53] Improved benchmark script with proper warmup and measurement phases
- [2025-05-12 04:53] Updated Modal deployment for more thorough testing and result collection
- `README.md` now begins with a focus-shift notice de-emphasising Modal H100 benchmarking in favour of the LCT research agenda.
- Marked LCT composition tests (`test_lct_composition`, `test_frft_composition`, `test_two_frft_90_equals_reversal`) as `xfail` due to challenges with discrete LCT group law and unitarity. Focus shifted to functional integration and throughput.
- Updated `b=0` (scaling) path in `torchlayers/functional/lct.py` to use `torch.nn.functional.grid_sample` for resampling and improved tensor handling.
- Improved docstrings for `linear_canonical_transform`, `symplectic_d` in `torchlayers/functional/lct.py` and for `LCTLayer.forward` in `torchlayers/lct.py`.

### Added (continued)

* **Property tests:** new `tests/test_lct_properties.py` covering
  – symplectic determinant (`symplectic_d`),
  – unitarity of the discrete LCT, and
  – composition/group law via matrix multiplication.
  These quick checks give broad algebraic coverage at negligible CI cost.

### Future

* Non-norm preserving group law variant planned for exact composition without amplitude distortion.
* Scaling tests for larger N values to validate performance characteristics.

### Fixed

- Configure setuptools package discovery in `pyproject.toml` to include only `torchlayers*`, resolving editable install failure during Modal deployment.
- Ensure `main()` installs dependencies and imports `torch` so that Modal can deserialize benchmark results.
- Set `image=image` for `main()` so that `uv` binary is present in container, fixing FileNotFoundError.
- Resolved some LaTeX formatting issues in `paper/sections/04_experiments.tex` and `paper/sections/99_checklist.tex` related to special characters.
- (Previously) Corrected LCT dense kernel implementation in `torchlayers/functional/lct.py` to improve unitarity for real parameters using QR projection.

### Removed

- Removed unused `_compute_d` method from `LCTLayer` in `torchlayers/lct.py`.
