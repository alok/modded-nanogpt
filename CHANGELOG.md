## [Unreleased]

### Added
- Cursor rule `.cursor/rules/neurips_2025_plan.mdc` tracks NeurIPS 2025 deliverables.
- Skeleton LCT layer (`torchlayers/lct.py`) with placeholder forward/inverse.
- Special-case test suite `tests/test_lct_special.py` (currently xfail) to cover Fourier/Laplace/Fresnel etc.
- Updated `AGENT.md` with new tasks and milestones.

### Changed
- Deprecated `.cursorrules` in favour of the canonical rule file.
- Refactored `tests/test_lct_special.py` to derive the Fourier reference via `torch.fft.fft`, cleaned up linter issues, and streamlined parametrisation.
