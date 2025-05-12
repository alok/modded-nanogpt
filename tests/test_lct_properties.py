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


def _random_symplectic(rng: random.Random) -> Tuple[float, float, float, float]:  # noqa: D401
    """Sample random real parameters (a,b,c,d) with ad − bc = 1 and b ≠ 0."""

    b = _rand_param(rng)
    a = _rand_param(rng)
    c = _rand_param(rng)
    d = symplectic_d(a, b, c)  # ensures determinant = 1
    return float(a), float(b), float(c), float(d)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("_", range(20))
def test_symplectic_determinant(_: int) -> None:
    """`symplectic_d` should satisfy ad − bc = 1 within tolerance."""

    rng = random.Random(_)
    a, b, c, d = _random_symplectic(rng)

    det = a * d - b * c
    assert math.isclose(det, 1.0, rel_tol=0.0, abs_tol=1e-6)


@pytest.mark.parametrize("_", range(10))
def test_lct_unitarity(_: int) -> None:
    """The LCT should be L²-norm preserving for real parameters with b ≠ 0."""

    rng = random.Random(42 + _)
    a, b, c, d = _random_symplectic(rng)

    N = 16
    x = torch.randn(N, dtype=torch.complex64)

    y = linear_canonical_transform(
        x,
        a=a,
        b=b,
        c=c,
        d=d,
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

    # Matrix multiplication M = M2 @ M1
    a = a2 * a1 + b2 * c1
    b = a2 * b1 + b2 * d1
    c = c2 * a1 + d2 * c1
    d = c2 * b1 + d2 * d1

    N = 8
    x = torch.randn(N, dtype=torch.complex64)

    # Sequential application
    out_seq = linear_canonical_transform(
        linear_canonical_transform(
            x,
            a=a1,
            b=b1,
            c=c1,
            d=d1,
            normalized=True,
            centered=True,
        ),
        a=a2,
        b=b2,
        c=c2,
        d=d2,
        normalized=True,
        centered=True,
    )

    # Single equivalent transform
    out_single = linear_canonical_transform(
        x,
        a=a,
        b=b,
        c=c,
        d=d,
        normalized=True,
        centered=True,
    )

    assert torch.allclose(out_seq, out_single, atol=1e-4), "Composition property failed." 