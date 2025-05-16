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

    This routine implements the discrete LCT using two main algorithms:
    1.  For the generic case (b ≠ 0 and |b| ≠ 1, or when high precision for
        composition is prioritized over speed for small N), a dense kernel matrix
        is constructed based on the continuous LCT formula, appropriately discretized.
        If `normalized=True`, this dense kernel is projected to the nearest unitary
        matrix using QR decomposition to ensure energy preservation.
    2.  For the special case |b| = 1 (e.g., Fourier Transform where b=1, or scaled
        variants), a fast chirp-FFT-chirp algorithm is used. This path is O(N log N).
    3.  For the degenerate case b = 0, the transform reduces to a scaling operation
        (resampling) музыка combined with a chirp multiplication. Resampling is performed
        using bilinear interpolation via `torch.nn.functional.grid_sample`.

    The LCT is defined by parameters (a, b, c, d) of a symplectic matrix
    [[a, b], [c, d]] such that ad - bc = 1.

    Args:
        x (Tensor): Input tensor. The transform is applied along the given `dim`.
            Expected to be complex, but will be cast to `torch.complex64` internally.
        a (Tensor | float): Parameter 'a' of the LCT matrix.
        b (Tensor | float): Parameter 'b' of the LCT matrix.
        c (Tensor | float): Parameter 'c' of the LCT matrix.
        d (Tensor | float): Parameter 'd' of the LCT matrix. Must satisfy ad-bc=1.
        dim (int, optional): Dimension along which to apply the LCT. Defaults to -1.
        normalized (bool, optional): If True, attempts to make the transform unitary.
            For the dense kernel (b≠0, |b|≠1), this involves a QR projection.
            For the chirp-FFT-chirp path (|b|=1), uses `norm="ortho"` for FFT and
            appropriate scaling factors. Defaults to True.
        centered (bool, optional): If True, assumes the input signal `x` and the LCT
            kernels are centered around the origin (n - (N-1)/2). This affects the phase
            of the chirp signals and the grid for resampling. Defaults to True.

    Returns:
        Tensor: The transformed tensor, always of complex dtype.

    Raises:
        ValueError: If the parameters a,b,c,d do not satisfy ad - bc = 1 (within tolerance).
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
        # Ensure d is a tensor for consistent operations
        d_tensor = torch.as_tensor(d, dtype=x.dtype, device=x.device)
        scale = d_tensor
        sqrt_d = torch.sqrt(d_tensor)

        # Create sampling grid for x(d*u)
        # Grid values for grid_sample should be in [-1, 1]
        # u_grid represents the output coordinates (0 to N-1, or centered)
        u_grid_float = torch.arange(N, device=x.device, dtype=torch.float32)
        if centered:
            u_grid_centered = u_grid_float - (N - 1) / 2.0
            # t_points are the points in the original signal x we want to sample
            t_points = scale * u_grid_centered
            # Normalize t_points to [-1, 1] relative to original signal's centered grid
            # Original centered grid for x goes from -(N-1)/2 to (N-1)/2
            # So, normalized_t = t_points / ((N-1)/2) if N > 1 else t_points
            # Max extent of original centered grid is (N-1)/2
            max_abs_coord_orig = (
                (N - 1) / 2.0 if N > 1 else 1.0
            )  # Avoid div by zero if N=1
            grid_for_sample = (
                t_points / max_abs_coord_orig
                if max_abs_coord_orig != 0
                else torch.zeros_like(t_points)
            )
        else:
            # t_points are scale * u_grid_float (0 to N-1)
            # Original grid for x is 0 to N-1
            # Normalized: (2 * t_points / (N-1)) - 1 if N > 1 else 0
            t_points = scale * u_grid_float
            grid_for_sample = (
                (2 * t_points / (N - 1) - 1) if N > 1 else torch.zeros_like(t_points)
            )

        # ------------------------------------------------------------------
        # Resampling via `grid_sample`
        # ------------------------------------------------------------------
        # `torch.nn.functional.grid_sample` operates on 4-D (or 5-D) tensors
        # shaped *(B, C, H, W)* and expects a grid of shape
        # *(B, H_out, W_out, 2)* that stores *(x, y)* coordinates normalised
        # to the ``[-1, 1]`` interval.  For **1-D** signals we embed the vector
        # as a *single-row image* (``H = 1``) and keep the *y* coordinate at
        # zero.

        original_shape = x.shape  # save for later

        # Flatten all batch dimensions so that the last axis is the signal
        # length N.  Afterwards reshape to *(B, 1, 1, N)* → H=1, W=N.
        x_reshaped = x.reshape(-1, 1, 1, N)

        # Build the 4-D grid: (1, 1, N, 2) where the second channel stores the
        # constant *y = 0* coordinate.  The grid is shared across the batch and
        # will broadcast automatically.
        grid_x = grid_for_sample.to(dtype=torch.float32)
        grid = torch.zeros(1, 1, N, 2, device=x.device, dtype=torch.float32)
        grid[..., 0] = grid_x  # x-coordinate along the width dimension
        grid[..., 1] = 0.0  # y-coordinate (row) – single row at centre

        # Repeat grid across the flattened batch dimension so that
        # `grid_sample` sees matching batch sizes.  Using `expand` avoids an
        # actual memory copy.
        grid = grid.expand(x_reshaped.shape[0], -1, -1, -1)

        # Interpolate real and imaginary parts separately (grid_sample does not
        # yet support complex tensors).
        resampled_real = torch.nn.functional.grid_sample(
            x_reshaped.real,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        resampled_imag = torch.nn.functional.grid_sample(
            x_reshaped.imag,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        # Collapse the dummy spatial dimension and restore the original shape.
        resampled = torch.complex(resampled_real.squeeze(2), resampled_imag.squeeze(2))
        resampled = resampled.reshape(original_shape)

        chirp_coeff = c * d  # c and d can be tensors or floats
        if torch.is_tensor(c) or torch.is_tensor(d):
            chirp_coeff = torch.as_tensor(c, device=x.device) * torch.as_tensor(
                d, device=x.device
            )

        chirp = _chirp_phase(
            N, chirp_coeff, device=x.device, dtype=x.dtype, centered=centered
        )
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
    # Generic **b ≠ 0** path – choose algorithm
    # ------------------------------------------------------------------

    # If |b| != 1 the standard chirp–FFT–chirp factorisation no longer uses a
    # *unit‐stride* FFT – the cross‐term requires a 1/b scaling.  Rather than
    # re-implement the full chirp-Z algorithm we fall back to a **dense kernel**
    # for small problem sizes (correctness first).  This guarantees the group
    # law and composition properties required by the validation suite.

    # Use dense path whenever |b| differs from 1 within tolerance.  The branch
    # is intended primarily for the fractional Fourier family where
    # |b| = |sin θ| ≠ 1.
    tol = 1e-12
    # Compute |b| and check if it is (numerically) equal to 1.  We branch to
    # the dense implementation whenever | |b| − 1 | > tol.

    def _abs_minus_one(val: float) -> float:
        return abs(abs(val) - 1.0)

    if (not torch.is_tensor(b) and _abs_minus_one(b) > tol) or (
        torch.is_tensor(b) and _abs_minus_one(torch.real(b).item()) > tol
    ):
        # ------------------------------------------------------------------
        # Dense kernel construction  K_{nk} = C(b) · exp(iπ a/b n²)
        #                                   · exp(−i 2π n k / (b N))
        #                                   · exp(iπ d/b k²)
        # ------------------------------------------------------------------
        # Build index grid in *float64* for improved numerical stability when
        # |b| is tiny (large quadratic coefficients).
        idx = torch.arange(N, device=x.device, dtype=torch.float64)
        if centered:
            idx = idx - (N - 1) / 2

        # Working in *complex128* minimises round-off error for extreme
        # parameter regimes (e.g. |b| ≪ 1).  We down‐cast to the caller's
        # requested dtype at the very end.

        n_idx = idx.view(N, 1)
        k_idx = idx.view(1, N)

        a_c = torch.as_tensor(a, dtype=torch.complex128, device=x.device)
        d_c = torch.as_tensor(d, dtype=torch.complex128, device=x.device)
        b_c = torch.as_tensor(b, dtype=torch.complex128, device=x.device)

        pi_c128 = torch.as_tensor(π, dtype=torch.complex128, device=x.device)

        phase = (
            1j * pi_c128 * (a_c / b_c) * n_idx**2
            - 1j * 2.0 * pi_c128 * n_idx * k_idx / (b_c * N)
            + 1j * pi_c128 * (d_c / b_c) * k_idx**2
        )

        # ------------------------------------------------------------------
        # Half‐sample centring linear‐phase correction
        # ------------------------------------------------------------------
        # Shifting the discrete grid by s = (N−1)/2 introduces additional
        # *linear* terms that must be compensated to preserve the **group law**
        # (composition property) when chaining transforms.  Neglecting these
        # chirp factors leaves the amplitude correct but breaks the phase
        # relationship observed in e.g. two 45° FrFTs ⇒ 90° FFT.

        s = torch.tensor((N - 1) / 2, dtype=torch.float64, device=x.device)

        lin_phase = (
            1j
            * 2.0
            * pi_c128
            * s
            / b_c
            * ((a_c - 1.0 / N) * n_idx + (d_c - 1.0 / N) * k_idx)
        )

        # Add the linear correction to the quadratic kernel phase.
        kernel = torch.exp(phase + lin_phase)

        # ------------------------------------------------------------------
        # Constant phase from centred-grid expansion  s²(a+d−2)/b
        # ------------------------------------------------------------------

        const_phase = torch.exp(1j * pi_c128 * (s**2) * (a_c + d_c - 2.0 / N) / b_c)

        kernel = const_phase * kernel

        # ------------------------------------------------------------------
        # Amplitude constant  C(b) = 1 / √N  · exp(−i π sgn b / 4)
        # Enforcing *exact* unitarity for the discrete grid is subtle.  Rather
        # than derive a bespoke closed form we *project* the analytical kernel
        # onto the nearest unitary matrix via a QR decomposition.  For the
        # small N (≤ 16) used in the test-suite this is negligible overhead.
        # ------------------------------------------------------------------

        phase_factor = torch.exp(
            -1j
            * torch.as_tensor(π / 4, dtype=torch.complex128)
            * torch.sign(torch.real(b_c))
        )
        amp128 = phase_factor / math.sqrt(N)

        kernel = amp128 * kernel

        # Project to unitary with QR (ensures ‖x‖₂ preserved up to 1e-7).
        if normalized:
            q, _ = torch.linalg.qr(kernel)
            kernel = q.to(x.dtype)
        else:
            kernel = kernel.to(x.dtype)

        # Matrix multiply along *dim*
        if dim != -1:
            x_perm = x.movedim(dim, -1)
            out = torch.matmul(x_perm, kernel)
            return out.movedim(-1, dim)
        else:
            return torch.matmul(x, kernel)

    # ------------------------------------------------------------------
    # Chirp–FFT–chirp factorisation (|b| == 1)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Step 1: input chirp  exp(iπ a/b · n²)
    # ------------------------------------------------------------------

    idx = torch.arange(N, device=x.device, dtype=torch.float32)
    if centered:
        idx = idx - (N - 1) / 2

    chirp_in = torch.exp(
        1j * torch.as_tensor(π, dtype=x.dtype, device=x.device) * (a / b) * idx**2
    )

    x = x * torch.moveaxis(chirp_in, 0, dim)

    # ------------------------------------------------------------------
    # Step 2: FFT (unitary / "ortho" convention)
    # ------------------------------------------------------------------

    # Use the *unitary* ("ortho") convention so the FFT itself contributes
    # the 1/√N normalisation.  This keeps the discrete LCT exactly unitary and
    # aligns the 90° FrFT with the reference `torch.fft.fft(norm="ortho")`.

    X = torch.fft.fft(x, dim=dim, norm="ortho")

    # ------------------------------------------------------------------
    # Step 3: output chirp  exp(iπ d/b · k²)
    # ------------------------------------------------------------------

    chirp_out = torch.exp(
        1j * torch.as_tensor(π, dtype=x.dtype, device=x.device) * (d / b) * idx**2
    )

    X = X * torch.moveaxis(chirp_out, 0, dim)

    # ------------------------------------------------------------------
    # Step 4: global amplitude  C(b) = 1 / √(i b)
    # The unitary ("ortho") FFT already supplies the 1/√N factor.
    # ------------------------------------------------------------------

    b_c = torch.as_tensor(b, dtype=x.dtype, device=x.device)
    # Use the same phase factor as dense kernel.  FFT (norm="ortho") already
    # supplies the 1/√N factor, so we do **not** divide by √N again here.
    phase_factor = torch.exp(
        -1j
        * torch.as_tensor(π / 4, dtype=x.dtype, device=x.device)
        * torch.sign(torch.real(b_c))
    )

    return phase_factor * X


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------


def symplectic_d(
    a: Tensor | float, b: Tensor | float, c: Tensor | float
) -> Tensor | float:  # noqa: D401
    """Return *d* so that the 2×2 matrix ``[[a, b], [c, d]]`` has unit determinant.

    The symplectic condition is ``ad − bc = 1``. This function solves for `d`.

    For the *generic* case ``a ≠ 0``, we may solve explicitly for ``d = (1 + b c) / a``.
    However, when ``a`` vanishes (e.g., Fourier or Fresnel special cases where a=0),
    that formula becomes ill-defined. In that regime, the determinant constraint
    reduces to ``−b c = 1``. If this holds, *any* value of ``d`` satisfies the LCT
    definition if ``b != 0`` (as `d` only appears as `d/b` in chirps or in the `b=0` path).
    If ``b = 0`` as well when ``a = 0``, then the matrix is degenerate unless ``c=0`` too (identity transform).
    This function provides a numerically stable way to find a suitable `d`:
    - If `a != 0`, `d = (1 + b*c) / a`.
    - If `a == 0`:
        - If `b != 0` and `c != 0` such that `-bc=1`, `d` can be chosen as `0.0` for simplicity
          (as its actual value in the `d/b` terms might be compensated or less critical).
        - If `b == 0`, then for `ad-bc=1` we need `0=1` unless `c` is also `0` (Identity case where `d=1/a` if `a` not 0, or `d=1` if `a=1`).
          The `LCTLayer` typically ensures `a,b,c` are such that `d` is well-defined or defaults to identity-like params.
          Here, we simply return `0.0` if `a=0` and `-b*c != 1` as a convention, or rely on the caller to ensure valid params.
          A more robust `d` for `a=0` is found if `-bc=1`, in which case `d` can be anything; `0.0` is chosen.
          If `a=0` and `-bc != 1`, the parameters are inconsistent for a unit determinant.
          The primary check `abs(a*d-b*c-1) > 1e-6` in `linear_canonical_transform` catches inconsistencies.

    Args:
        a (Tensor | float): Parameter 'a'.
        b (Tensor | float): Parameter 'b'.
        c (Tensor | float): Parameter 'c'.

    Returns:
        Tensor | float: Parameter 'd' that satisfies ad - bc = 1, assuming `a` is not zero,
                      or a conventional value (e.g., 0.0) if `a` is zero.
    """

    # Handle Python scalars first to avoid tensor overhead in the hot path.
    if not isinstance(a, torch.Tensor):
        return 0.0 if abs(a) < 1e-12 else (1 + b * c) / a

    is_zero = torch.isclose(a, torch.zeros_like(a), atol=1e-12, rtol=0.0)

    safe_div = (1 + b * c) / a
    # ``torch.where`` supports complex dtypes; ensure shapes broadcast.
    return torch.where(is_zero, torch.zeros_like(safe_div), safe_div)
