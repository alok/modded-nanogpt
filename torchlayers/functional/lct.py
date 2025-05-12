"""Numerical kernels for the **1-D discrete Linear Canonical Transform**.

This file hosts the low-level, batch-broadcastable primitives.  The public
`LCTLayer` class wraps these functions and adds parameter management.

The current implementation prioritises *clarity & correctness* over raw speed
but is easily torch-compiled later.
"""

from __future__ import annotations

import math
from typing import Final

import torch

Tensor = torch.Tensor

π: Final[float] = math.pi

__all__ = [
    "linear_canonical_transform",
    "symplectic_d",
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _chirp_phase(
    length: int,
    coeff: Tensor | float,
    /,
    *,
    device: torch.device,
    dtype: torch.dtype,
    centered: bool = True,
) -> Tensor:
    """Return `exp( i π * coeff * n² )` as 1-D tensor of shape *(length,)*."""

    n = torch.arange(length, device=device, dtype=dtype)
    if centered:
        n = n - length // 2
    phase = 1j * π * coeff * n**2
    return torch.exp(phase)


# -----------------------------------------------------------------------------
# Public: discrete LCT kernel
# -----------------------------------------------------------------------------


def linear_canonical_transform(
    x: Tensor,
    *,
    a: Tensor | float,
    b: Tensor | float,
    c: Tensor | float,
    d: Tensor | float,
    dim: int = -1,
    normalized: bool = True,
    centered: bool = True,
) -> Tensor:
    """Apply the discrete Linear Canonical Transform along *dim*.

    This routine implements the **chirp–FFT–chirp** factorisation and supports
    *complex64/128* inputs.  It returns a complex tensor regardless of the
    input dtype.
    """

    if abs(a * d - b * c - 1) > 1e-6:
        raise ValueError("LCT parameters must satisfy ad − bc = 1.")

    x = x.to(torch.complex64)
    N = x.size(dim)

    if b == 0:  # degenerate scaling branch
        scale = d
        sqrt_d = torch.sqrt(torch.tensor(d, dtype=x.dtype, device=x.device))

        idx = torch.arange(N, device=x.device, dtype=torch.float32)
        if centered:
            idx = idx - N // 2

        # Simple nearest-neighbour resample for MVP; refine later.
        sample_pos = (scale * idx).round().clamp(0, N - 1).to(torch.long)
        resampled = x.index_select(dim, sample_pos)

        chirp = _chirp_phase(N, c * d, device=x.device, dtype=x.dtype, centered=centered)
        resampled = sqrt_d * resampled * torch.moveaxis(chirp, 0, dim)
        return resampled

    # Generic b ≠ 0 path ----------------------------------------------------

    coeff_in = a / b
    coeff_out = d / b

    chirp_in = _chirp_phase(N, coeff_in, device=x.device, dtype=x.dtype, centered=centered)
    chirp_out = _chirp_phase(N, coeff_out, device=x.device, dtype=x.dtype, centered=centered)

    x = x * torch.moveaxis(chirp_in, 0, dim)

    norm_mode = "ortho" if normalized else "backward"
    X = torch.fft.fft(x, dim=dim, norm=norm_mode)

    X = X * torch.moveaxis(chirp_out, 0, dim)

    const = 1.0 / torch.sqrt(1j * torch.tensor(b, dtype=x.dtype, device=x.device))
    return const * X


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------


def symplectic_d(a: Tensor | float, b: Tensor | float, c: Tensor | float) -> Tensor | float:  # noqa: D401
    """Return *d* so that [[a, b], [c, d]] ∈ SL(2,ℂ)."""

    return (1 + b * c) / a
