"""Property‐based tests for additional Linear Canonical Transform invariants.

These tests capture a *Pareto* subset of algebraic properties that provide high
coverage at relatively low implementation effort:

1. **Symplectic determinant** – `symplectic_d` must ensure `ad − bc = 1`.
2. **Unitarity** – for real parameters with `b ≠ 0` the LCT is energy‐
   preserving (‖x‖₂ = ‖LCT(x)‖₂).
3. **Composition (group law)** – sequential LCTs correspond to matrix
   multiplication of their parameter matrices.

All tests run on CPU with small problem sizes to keep CI fast.
"""

from __future__ import annotations

import math
import pathlib
import random
import sys
from typing import Tuple

import pytest
import torch

# Ensure project root is importable before third-party imports
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torchlayers.functional.lct import (
    linear_canonical_transform,
    symplectic_d,
)

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------


def _rand_param(rng: random.Random) -> float:
    """Return a random float in a moderate range avoiding zero."""

    val = 0.0
    while abs(val) < 1e-3:  # keep away from ill-conditioned zero
        val = rng.uniform(-2.0, 2.0)
    return val


def _random_symplectic(
    rng: random.Random,
    *,
    real_only: bool = False,
) -> Tuple[complex, complex, complex, complex]:  # noqa: D401
    """Sample **complex** parameters ``(a,b,c,d)`` with the unimodular constraint.

    Args
    ----
    rng
        Random-number generator instance.
    real_only
        When *True*, imaginary parts are forced to **zero** (legacy behaviour).

    Notes
    -----
    Keeping a flag avoids littering call-sites with post-hoc casts.  The new
    default exercises the full complex path so edge cases receive coverage.
    """

    def _rand_complex_param() -> complex:
        real = _rand_param(rng)
        imag = 0.0 if real_only else _rand_param(rng)
        return complex(real, imag)

    b: complex = _rand_complex_param()
    a: complex = _rand_complex_param()
    c: complex = _rand_complex_param()

    d: complex = symplectic_d(a, b, c)  # type: ignore[assignment]

    return a, b, c, d


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("_", range(20))
def test_symplectic_determinant(_: int) -> None:
    """`symplectic_d` should satisfy ad − bc = 1 within tolerance."""

    rng = random.Random(_)
    a, b, c, d = _random_symplectic(rng)

    det = a * d - b * c
    # For complex values we compare the *magnitude* of the deviation from 1.
    assert abs(det - 1.0) < 1e-6


@pytest.mark.parametrize("_", range(10))
def test_lct_unitarity(_: int) -> None:
    """The LCT should be L²-norm preserving for real parameters with b ≠ 0."""

    rng = random.Random(42 + _)
    # Unitarity holds generically **only for real parameters**.  Force
    # imaginary components zero for this specific test.
    a, b, c, d = _random_symplectic(rng, real_only=True)

    N = 16
    x = torch.randn(N, dtype=torch.complex64)

    # Convert parameters to tensors so type checker is satisfied.
    a_t, b_t, c_t, d_t = (
        torch.as_tensor(a, dtype=torch.complex64),
        torch.as_tensor(b, dtype=torch.complex64),
        torch.as_tensor(c, dtype=torch.complex64),
        torch.as_tensor(d, dtype=torch.complex64),
    )

    y = linear_canonical_transform(
        x,
        a=a_t,
        b=b_t,
        c=c_t,
        d=d_t,
        normalized=True,
        centered=True,
    )

    assert torch.allclose(x.norm(), y.norm(), atol=1e-4)


@pytest.mark.parametrize("_", range(5))
def test_lct_composition(_: int) -> None:
    """Composition law: L2 ∘ L1 == L where M = M2·M1."""

    rng = random.Random(1234 + _)

    # First transform (a1,b1,c1,d1)
    a1, b1, c1, d1 = _random_symplectic(rng)

    # Second transform (a2,b2,c2,d2)
    a2, b2, c2, d2 = _random_symplectic(rng)
    
    # Stack parameters into 2x2 matrices
    M1 = torch.tensor([[a1, b1], [c1, d1]], dtype=torch.complex128)
    M2 = torch.tensor([[a2, b2], [c2, d2]], dtype=torch.complex128)
    
    # Matrix multiplication M = M2 @ M1
    M = torch.matmul(M2, M1)
    a, b, c, d = M[0, 0].item(), M[0, 1].item(), M[1, 0].item(), M[1, 1].item()

    N = 8
    x = torch.randn(N, dtype=torch.complex64)

    # Sequential application
    out_seq = linear_canonical_transform(
        linear_canonical_transform(
            x,
            a=torch.as_tensor(a1, dtype=torch.complex64),
            b=torch.as_tensor(b1, dtype=torch.complex64),
            c=torch.as_tensor(c1, dtype=torch.complex64),
            d=torch.as_tensor(d1, dtype=torch.complex64),
            normalized=True,
            centered=True,
        ),
        a=torch.as_tensor(a2, dtype=torch.complex64),
        b=torch.as_tensor(b2, dtype=torch.complex64),
        c=torch.as_tensor(c2, dtype=torch.complex64),
        d=torch.as_tensor(d2, dtype=torch.complex64),
        normalized=True,
        centered=True,
    )

    # Single equivalent transform
    out_single = linear_canonical_transform(
        x,
        a=torch.as_tensor(a, dtype=torch.complex64),
        b=torch.as_tensor(b, dtype=torch.complex64),
        c=torch.as_tensor(c, dtype=torch.complex64),
        d=torch.as_tensor(d, dtype=torch.complex64),
        normalized=True,
        centered=True,
    )

    assert torch.allclose(out_seq, out_single, atol=1e-4), "Composition property failed." 