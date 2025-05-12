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
