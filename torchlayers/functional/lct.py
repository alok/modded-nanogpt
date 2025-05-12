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

    # torch.arange does **not** support complex dtypes, so we build the phase
    # factor explicitly from real components and cast at the end.

    n = torch.arange(length, device=device, dtype=torch.float32)
    if centered:
        n = n - length // 2

    # Imaginary component of the exponent: π * coeff * n²
    # (Real part is zero.)  Supports autograd when *coeff* is a learnable
    # tensor.
    imag = torch.tensor(π, device=device, dtype=torch.float32) * coeff * n**2

    phase = torch.complex(torch.zeros_like(imag), imag.to(torch.float32))
    return torch.exp(phase).to(dtype)


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
    """Return *d* so that the 2×2 matrix ``[[a, b], [c, d]]`` has unit determinant.

    The symplectic condition is ``ad − bc = 1``.  For the *generic* case
    ``a ≠ 0`` we may solve explicitly for ``d = (1 + b c) / a``.  However, when
    ``a`` vanishes (e.g. Fourier–Fresnel special cases) that formula becomes
    ill-defined.  In that regime the determinant constraint reduces to
    ``−b c = 1`` and **any** value of ``d`` satisfies the requirement.  We
    choose the minimal solution ``d = 0`` for numerical stability.
    """

    # Handle Python scalars first to avoid tensor overhead in the hot path.
    if not isinstance(a, torch.Tensor):
        return 0.0 if abs(a) < 1e-12 else (1 + b * c) / a

    is_zero = torch.isclose(a, torch.zeros_like(a), atol=1e-12, rtol=0.0)

    safe_div = (1 + b * c) / a
    # ``torch.where`` supports complex dtypes; ensure shapes broadcast.
    return torch.where(is_zero, torch.zeros_like(safe_div), safe_div)
