from __future__ import annotations

import math
from typing import Tuple

import pytest
import torch
from jaxtyping import Complex, Float

from torchlayers.functional.lct import linear_canonical_transform as lct, symplectic_d

π: float = math.pi

def frft_params(theta: float) -> Tuple[float, float, float, float]:
    """Return (a, b, c, d) parameters for a fractional Fourier transform of angle *theta*.

    Continuous-time definition:
        (a,b,c,d) = (cos θ, sin θ, −sin θ, cos θ).
    """
    c, s = math.cos(theta), math.sin(theta)
    return c, s, -s, c


@pytest.mark.parametrize("theta", [π / 4])
@pytest.mark.parametrize("N", [64, 128])
def test_frft_composition(theta: float, N: int) -> None:
    """Two 45° FrFTs must equal one 90° FrFT (i.e. the FFT) up to a global phase."""

    x: Complex[torch.Tensor, "N"] = torch.randn(N, dtype=torch.float32) + 1j * torch.randn(N, dtype=torch.float32)

    a, b, c, d = frft_params(theta)
    y_seq = lct(lct(x, a=a, b=b, c=c, d=d), a=a, b=b, c=c, d=d)

    # Single 2θ transform (90°)
    a2, b2, c2, d2 = frft_params(2.0 * theta)
    y_single = lct(x, a=a2, b=b2, c=c2, d=d2)

    # Allow a global complex phase (extract from first entry)
    phase = y_seq[0] / y_single[0]
    assert torch.allclose(y_seq, phase * y_single, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("N", [32, 64])
def test_unitarity_negative_b(N: int) -> None:
    """Random (a,b,c) with negative real b should still yield a unitary transform."""

    rng = torch.Generator().manual_seed(0)

    # Generate random real a, negative real b, random real c
    a = torch.randn((), generator=rng).item()
    b = -torch.rand((), generator=rng).item() - 0.1  # ensure < 0
    c = torch.randn((), generator=rng).item()
    d = symplectic_d(a, b, c)

    x: Complex[torch.Tensor, "N"] = torch.randn(N, generator=rng) + 1j * torch.randn(N, generator=rng)

    y = lct(x, a=a, b=b, c=c, d=d)

    norm_in: Float[torch.Tensor, ""] = torch.linalg.norm(x)
    norm_out: Float[torch.Tensor, ""] = torch.linalg.norm(y)

    assert torch.allclose(norm_in, norm_out, atol=1e-6, rtol=1e-6) 