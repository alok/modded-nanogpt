"""Core smoke-tests for :pyclass:`torchlayers.lct.LCTLayer`.

Validates two fundamental properties expected from a *working* Linear Canonical
Transform implementation:

1. **Fourier reduction** – when `(a,b,c) = (0,1,0)` the LCT must coincide with
   the discrete FFT (unitary 2π-conjugate convention).
2. **Inverse consistency** – applying the inverse immediately after the forward
   recovers the original signal to numerical precision.
"""

from __future__ import annotations

import sys
import pathlib

# Ensure project root is importable *before* third-party imports
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch

from torchlayers.lct import LCTLayer

# -----------------------------------------------------------------------------
# Fixtures & helpers
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def random_signal() -> torch.Tensor:  # noqa: D401
    """Return a small random complex vector for testing."""

    torch.manual_seed(0)
    return torch.randn(8, dtype=torch.complex64)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_fft_reduction(random_signal: torch.Tensor):
    """When (a,b,c)=(0,1,0) the LCT should equal the FFT."""

    sig = random_signal
    layer = LCTLayer(a=0.0, b=1.0, c=0.0, normalized=True)

    out = layer(sig)
    expected = torch.fft.fft(sig, norm="ortho")

    assert torch.allclose(out, expected, atol=1e-6)


def test_inverse_identity(random_signal: torch.Tensor):
    """Applying forward followed by inverse should reconstruct the input."""

    layer = LCTLayer(a=0.3, b=1.0, c=-0.1, normalized=True)
    recon = layer.inverse(layer(random_signal))

    assert torch.allclose(recon, random_signal, atol=1e-6)


# -----------------------------------------------------------------------------
# New tests – degenerate b = 0 branch (pure scaling + phase)
# -----------------------------------------------------------------------------


def test_b_zero_scaling_branch():  # noqa: D401
    """LCT with *b = 0* followed by its inverse should be identity.

    Choose parameters ``(a,b,c,d) = (1,0,2,1)`` which satisfy the symplectic
    constraint ``ad − bc = 1``.  The forward path exercises the dedicated
    scaling branch inside :pyfunc:`linear_canonical_transform`.
    """

    a, b, c = 1.0, 0.0, 2.0  # d will be computed internally (→ 1.0)

    torch.manual_seed(42)
    x = torch.randn(16, dtype=torch.complex64)

    layer = LCTLayer(a=a, b=b, c=c, normalized=True)

    y = layer(x)
    recon = layer.inverse(y)

    assert torch.allclose(recon, x, atol=1e-5)
