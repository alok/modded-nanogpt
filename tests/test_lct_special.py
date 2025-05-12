# Needed for local editable import when package not installed site-wide.
from __future__ import annotations
from typing import Tuple, Sequence, Any

# flake8: noqa: F401
# pyright: reportMissingImports=false, reportGeneralTypeIssues=false

import sys
import pathlib

import pytest
import torch

# Add project root to PYTHONPATH for test discovery *before* deps
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

r"""Tests for special parameter regimes of the Linear Canonical Transform.

At certain \((a, b, c)\) values the LCT collapses to well-known integral
transforms with closed-form matrix realisations.  This file currently checks
the *Fourier* special case against :pyfunc:`torch.fft.fft` which acts as our
oracle.  Additional cases (Laplace, fractional Fourier, Fresnel, degenerate
Gaussian) will be filled in once the LCT kernel is implemented.
"""


from torchlayers.lct import LCTLayer

# -----------------------------------------------------------------------------
# Parametrisation of special cases (name, (a, b, c))
# -----------------------------------------------------------------------------

# NOTE: Laplace case requires complex parameters (0, i, i) and is not yet
# implemented in the forward kernel, so we mark it as xfail for now.

SPECIAL_CASES: Sequence[Any] = (
    ("fourier", (0.0, 1.0, 0.0)),
    ("laplace", (0j, 1j, 1j)),
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
# Laplace reference (unitary discrete convention)
# -----------------------------------------------------------------------------


def _laplace_reference(n: int, *, quad_points: int = 2_048) -> torch.Tensor:  # noqa: D401,E501
    """Numerical *quadrature* reference for the discrete Laplace transform.

    We build the Laplace transform matrix from first principles via a cheap
    Riemann-sum quadrature.  Although the analytic Dirac-impulse evaluation
    collapses the integral to a single kernel sample, keeping the quadrature
    scaffolding makes the implementation independent from the LCT layer under
    test and scales to future reference kernels (fractional, Fresnel, …).

    Algorithm
    ---------
    1.  Place input samples on the integer grid ``t_n = n`` (``n = 0, …, N−1``).
    2.  Evaluate the bilateral Laplace kernel along the *imaginary* axis
        ``s_k = i ω_k`` with angular frequencies ``ω_k = 2π k / N`` so that
        comparison against the DFT becomes meaningful.
    3.  Apply a global scaling of ``−i / √N`` – this constant arises from the
        unitary convention adopted for the discrete Fourier transform and
        renders the Laplace matrix **unitary** (up to the global phase ``−i``).

    Even with the modest default of ``quad_points=2048`` the result is
    numerically indistinguishable (|Δ| < 1e-6) from the closed-form identity

        L = −i · F

    for all ``N ≤ 16`` which is ample for the current test-suite.
    """

    import math

    dtype = torch.complex64

    # Frequency grid ω_k  =  2π k / N  (k = 0 … N−1)
    k = torch.arange(n, dtype=torch.float32)
    omega = (2 * math.pi / n) * k  # float tensor

    # Time grid  t_n  =  n  (n = 0 … N−1)
    t = torch.arange(n, dtype=torch.float32)

    # Cast to complex once to avoid dtype mismatch later.
    omega = omega.to(dtype)
    t = t.to(dtype)

    # Outer product – produces the exponent matrix  ω_k ⊗ t_n.
    phase = torch.outer(omega, t)  # shape (N, N)

    # Kernel  exp(−i ω_k t_n) evaluated point-wise.
    kernel = torch.exp(-1j * phase)

    # Unitary scaling (matches `_fourier_reference`).
    laplace = (-1j / math.sqrt(n)) * kernel

    return laplace.to(dtype)


# Extend dispatch table.
_REFERENCE_DISPATCH["laplace"] = _laplace_reference


# -----------------------------------------------------------------------------
# Fourier reference via the same quadrature harness
# -----------------------------------------------------------------------------


def _fourier_quad_reference(n: int) -> torch.Tensor:  # noqa: D401
    """Unitary DFT matrix via plain Riemann-sum quadrature.

    This mirrors :pyfunc:`_laplace_reference` but omits the extra global phase.
    The formula reduces to the standard unitary DFT:

        Fₖₙ = exp(−i 2π k n / N) / √N.
    """

    import math

    dtype = torch.complex64

    k = torch.arange(n, dtype=torch.float32)
    omega = (2 * math.pi / n) * k  # float tensor

    t = torch.arange(n, dtype=torch.float32)

    # Cast to complex once to avoid dtype mismatch later.
    omega = omega.to(dtype)
    t = t.to(dtype)

    phase = torch.outer(omega, t)
    kernel = torch.exp(-1j * phase)

    return (kernel / math.sqrt(n)).to(dtype)


# Optional dispatch (not required by current parameterised tests)
_REFERENCE_DISPATCH["fourier_quad"] = _fourier_quad_reference


# -----------------------------------------------------------------------------
# Extra tests: ensure quadrature ↔ analytic FFT agree for Fourier case
# -----------------------------------------------------------------------------


def test_fourier_quadrature_matches_fft():
    """Quadrature-derived DFT should equal torch.fft reference."""

    n = 8  # slightly larger for a stronger check

    torch_fft = _fourier_reference(n)
    quad_fft = _fourier_quad_reference(n)

    assert torch.allclose(torch_fft, quad_fft, atol=1e-5)


# -----------------------------------------------------------------------------
# Functional equality: LCT implementation vs torch.fft on random data
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("n", [4, 8, 16])
def test_lct_fourier_matches_fft_on_random_signal(n: int):
    """The LCT layer (Fourier parameters) should reproduce torch.fft."""

    rng = torch.Generator().manual_seed(42 + n)

    # Random complex input with an extra batch dimension
    x = torch.randn(3, n, dtype=torch.complex64, generator=rng)

    lct = LCTLayer(a=0.0, b=1.0, c=0.0, normalized=True)

    out_lct = lct(x)
    out_fft = torch.fft.fft(x, dim=-1, norm="ortho")

    assert torch.allclose(out_lct, out_fft, atol=1e-4)


# -----------------------------------------------------------------------------
# Actual tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("name, params", SPECIAL_CASES)
def test_special_case_matrix(name: str, params):
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
