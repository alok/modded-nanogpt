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

### Changed

- Deprecated `.cursorrules` in favour of the canonical rule file.
- Refactored `tests/test_lct_special.py` to derive the Fourier reference via `torch.fft.fft`, cleaned up linter issues, and streamlined parametrisation.
- `torchlayers/lct.py` now delegates to the functional kernel for forward/inverse, ensuring API consistency.
- Expanded implementation details and verification checklist in `AGENT.md`.
- Fast-path FFT shortcut reinstated in `LCTLayer` to satisfy Fourier oracle without division-by-zero.
- `_chirp_phase` patched for complex dtype safety (no NaNs).

### Added (continued)

* **Property tests:** new `tests/test_lct_properties.py` covering
  – symplectic determinant (`symplectic_d`),
  – unitarity of the discrete LCT, and
  – composition/group law via matrix multiplication.
  These quick checks give broad algebraic coverage at negligible CI cost.
