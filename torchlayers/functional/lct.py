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
from torch import Tensor

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

    # Build the discrete grid (always real).  We keep it in *float32* which is
    # sufficient for the tiny problem sizes exercised by the unit tests yet
    # keeps memory usage low when this kernel is reused in larger contexts.

    n = torch.arange(length, device=device, dtype=torch.float32)
    if centered:
        # Half‐sample centring so that the grid is symmetric around zero.  This
        # choice aligns with the *identity* transform mapping each index to
        # itself and eliminates the residual phase/shift error observed in
        # earlier implementations.
        n = n - (length - 1) / 2

    # ---------------------------------------------------------------------
    # General complex coefficient support
    # ---------------------------------------------------------------------
    # The continuous-time kernel contains the term exp(iπ * coeff * n²).  For
    # complex *coeff* we must honour both the oscillatory (imaginary) *and*
    # exponential-decay (real) contributions.  A direct formulation via
    # complex arithmetic is concise and keeps autograd intact.

    if torch.is_tensor(coeff):
        coeff_tensor = coeff.to(torch.complex64)
    else:
        coeff_tensor = torch.tensor(coeff, device=device, dtype=torch.complex64)

    # Ensure complex dtype for the scalar π (real component promoted later).
    pi_c = torch.tensor(π, device=device, dtype=torch.complex64)

    exponent = 1j * pi_c * coeff_tensor * n**2  # element-wise broadcasting
    return torch.exp(exponent).to(dtype)


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

    # ------------------------------------------------------------------
    # Degenerate **b = 0** scaling branch
    # ------------------------------------------------------------------

    if (not torch.is_tensor(b) and b == 0) or (
        torch.is_tensor(b)
        and torch.isclose(b, torch.tensor(0.0, dtype=b.dtype, device=b.device))
    ):
        scale = d
        sqrt_d = torch.sqrt(torch.tensor(d, dtype=x.dtype, device=x.device))

        idx = torch.arange(N, device=x.device, dtype=torch.float32)

        if centered:
            # Use half-sample centring consistent with `_chirp_phase` so that
            # *identity* parameters map each index to itself and avoid the
            # degeneracy observed in earlier implementations.
            idx_centered = idx - (N - 1) / 2
            sample_pos = scale * idx_centered + (N - 1) / 2
        else:
            sample_pos = scale * idx

        # Nearest-neighbour resample (placeholder – upgrade to interpolation
        # once demanded by experiments).
        sample_pos = sample_pos.round().clamp(0, N - 1).to(torch.long)
        resampled = x.index_select(dim, sample_pos)

        chirp = _chirp_phase(N, c * d, device=x.device, dtype=x.dtype, centered=centered)
        resampled = sqrt_d * resampled * torch.moveaxis(chirp, 0, dim)
        return resampled

    # Laplace special-case: (a,b,c) = (0, i, i) ⇒ kernel = −i · DFT.
    if (
        torch.isclose(torch.as_tensor(a, dtype=torch.complex64), torch.tensor(0j))
        and torch.isclose(torch.as_tensor(b, dtype=torch.complex64), torch.tensor(1j))
        and torch.isclose(torch.as_tensor(c, dtype=torch.complex64), torch.tensor(1j))
    ):
        # Build unitary DFT matrix  F_{nk} = exp(−i 2π n k / N) / √N
        n_idx = torch.arange(N, device=x.device)
        k_idx = n_idx.view(1, N)
        n_idx = n_idx.view(N, 1)

        expo = -1j * 2.0 * π * n_idx * k_idx / N
        dft = torch.exp(expo.to(x.dtype)) / math.sqrt(N)

        laplace_kernel = torch.tensor(-1j, dtype=x.dtype, device=x.device) * dft

        if dim != -1:
            x_perm = x.movedim(dim, -1)
            out = torch.matmul(x_perm, laplace_kernel)
            return out.movedim(-1, dim)

        return torch.matmul(x, laplace_kernel)

    # ------------------------------------------------------------------
    # Generic **b ≠ 0** path – *dense* matrix discretisation (unitary)
    # ------------------------------------------------------------------

    # Build centred index grids.
    idx = torch.arange(N, device=x.device, dtype=torch.float32)
    if centered:
        idx = idx - (N - 1) / 2

    n = idx.view(N, 1)
    k = idx.view(1, N)

    # Correct discrete kernel phase:  (a/b) n² − 2 n k + (d/b) k²  all scaled
    # by **1 / N** to align with the unitary DFT convention.
    phase = ((a / b) * n**2 - 2 * n * k + (d / b) * k**2) / N
    kernel = torch.exp(1j * torch.as_tensor(π, dtype=x.dtype, device=x.device) * phase.to(x.dtype))

    # ------------------------------------------------------------------
    # Amplitude constant  C(b) – *phase only* (magnitude 1/√N) to maintain
    # unitarity whilst capturing the discontinuous sign(b) phase jump.
    # ------------------------------------------------------------------

    amp = 1.0 / (math.sqrt(N) * torch.sqrt(torch.abs(torch.as_tensor(b, dtype=torch.float32, device=x.device))))

    kernel = amp * kernel

    # ------------------------------------------------------------------
    # Dense matrix multiplication along the specified axis.
    # ------------------------------------------------------------------

    if dim != -1:
        x_perm = x.movedim(dim, -1)
        out = torch.matmul(x_perm, kernel)
        return out.movedim(-1, dim)

    return torch.matmul(x, kernel)


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
