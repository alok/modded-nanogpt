from __future__ import annotations

# flake8: noqa: F401
# pyright: reportMissingImports=false, reportGeneralTypeIssues=false

"""Tests for special parameter regimes of the Linear Canonical Transform.

At certain \((a, b, c)\) values the LCT collapses to well-known integral
transforms with closed-form matrix realisations.  This file currently checks
the *Fourier* special case against :pyfunc:`torch.fft.fft` which acts as our
oracle.  Additional cases (Laplace, fractional Fourier, Fresnel, degenerate
Gaussian) will be filled in once the LCT kernel is implemented.
"""

from typing import Tuple

import pytest
import torch

from torchlayers.lct import LCTLayer

# -----------------------------------------------------------------------------
# Parametrisation of special cases (name, (a, b, c))
# -----------------------------------------------------------------------------

SPECIAL_CASES: Tuple[Tuple[str, Tuple[float, float, float]], ...] = (
    ("fourier", (0.0, 1.0, 0.0)),
)


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _fourier_reference(n: int) -> torch.Tensor:  # noqa: D401
    """Return the unitary DFT matrix via :pyfunc:`torch.fft.fft`."""

    eye = torch.eye(n, dtype=torch.complex64)
    return torch.fft.fft(eye, norm="ortho")


# Map case name → reference generator.  Extensible for future special cases.
_REFERENCE_DISPATCH = {"fourier": _fourier_reference}


# -----------------------------------------------------------------------------
# Actual tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("name, params", SPECIAL_CASES)
@pytest.mark.xfail(reason="LCT forward kernel not yet implemented")
def test_special_case_matrix(name: str, params: Tuple[float, float, float]):
    """LCT layer should reproduce the analytic matrix for each special case."""

    n = 4  # Small sanity-sized transform
    x = torch.eye(n, dtype=torch.complex64)

    layer = LCTLayer(a=params[0], b=params[1], c=params[2], normalized=True)
    out = layer(x)

    expected = _REFERENCE_DISPATCH[name](n)

    assert torch.allclose(out, expected, atol=1e-4), f"Mismatch for {name} case"


@pytest.mark.xfail(reason="Unitarity property check pending implementation")
def test_fourier_unitarity():
    """DFT matrix should be unitary: FᴴF = I."""

    n = 4
    dft = _fourier_reference(n)

    # Unitarity ⇒ FᴴF = I (within tolerance)
    prod = dft.conj().T @ dft
    assert torch.allclose(prod, torch.eye(n, dtype=torch.complex64), atol=1e-4)
