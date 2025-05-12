from __future__ import annotations

"""Linear Canonical Transform (LCT) layer skeleton.

This is a placeholder implementation that exposes the public API expected by
upcoming tests.  The mathematical core will be filled in later.

Parameters (a, b, c) follow the symplectic matrix parameterisation:
S = [[a, b], [c, d]] with ad - bc = 1.  For now we store only (a, b, c) and
compute d on the fly once the layer is implemented.

`normalized=True` selects the *unitary* 2π-conjugate convention analogous to
NumPy's `np.fft.fft(..., norm="ortho")`.
"""

from typing import Optional

import torch
from torch import nn

# Type aliases for readability
Tensor = torch.Tensor

__all__ = ["LCTLayer"]


class LCTLayer(nn.Module):
    """Differentiable Linear Canonical Transform (skeleton).

    Args:
        a, b, c: Real scalars initialising the symplectic parameters.
        normalized: If True (default) use unitary 2π-conjugate convention.
    """

    def __init__(
        self,
        *,
        a: complex | float = 0.0,
        b: complex | float = 1.0,
        c: complex | float = 0.0,
        normalized: bool = True,
    ) -> None:  # noqa: D401,E501
        super().__init__()

        def _to_param(val: complex | float) -> nn.Parameter:  # helper
            dtype = torch.complex64 if isinstance(val, complex) else torch.float32
            return nn.Parameter(torch.tensor(val, dtype=dtype))

        self.a = _to_param(a)
        self.b = _to_param(b)
        self.c = _to_param(c)
        self.normalized = normalized

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        """Apply the Linear Canonical Transform (minimal forward pass).

        This MVP kernel realises the *chirp–FFT–chirp* factorisation for the
        generic (a, b, c) parameter regime with **b ≠ 0**.  The implementation
        is deliberately limited to the forward direction because it is all we
        need for the micro-benchmark and Fourier-special-case unit tests.

        Notes
        -----
        1. The input is expected to be complex-valued with the transform axis
           residing on the *last* dimension so that broadcasting works for any
           batch shape.
        2. When ``normalized`` is *True* we request ``norm="ortho"`` from
           ``torch.fft.fft`` which already applies the 1 / √N factor.  When
           ``normalized`` is *False* we mimic NumPy/PyTorch default FFT
           semantics and leave scaling untouched.
        3. Degenerate **b = 0** branch (pure scaling + phase) will be handled
           later once required by additional tests.
        """

        import math  # local to avoid polluting module namespace

        if not torch.is_complex(x):
            raise TypeError("LCTLayer expects a complex-valued input tensor.")

        N = x.size(-1)
        if N == 0:
            return x  # trivial

        # Ensure *scalar* parameters are plain Python floats for math ops.
        a: Tensor = self.a.squeeze()
        b: Tensor = self.b.squeeze()
        c: Tensor = self.c.squeeze()

        # Handle pure Fourier case quickly: (a=0, c=0, b≠0)
        zero_a = torch.tensor(0.0, dtype=a.dtype, device=a.device)
        zero_b = torch.tensor(0.0, dtype=b.dtype, device=b.device)
        zero_c = torch.tensor(0.0, dtype=c.dtype, device=c.device)

        if torch.isclose(a, zero_a) and torch.isclose(c, zero_c):
            norm_mode = "ortho" if self.normalized else "backward"
            return torch.fft.fft(x, dim=-1, norm=norm_mode)

        if torch.isclose(b, zero_b):
            raise NotImplementedError("LCT forward kernel for b = 0 not yet implemented.")

        # Prepare index range along the transform axis: 0, 1, …, N-1 (real dtype)
        idx = torch.arange(N, device=x.device, dtype=x.real.dtype)

        # First chirp ϕ₁(n) = exp(i·π·a·n² / (N·b))
        phi1 = torch.exp(1j * math.pi * a * (idx ** 2) / (N * b))

        # Symplectic condition ⇒ d = (1 + b·c) / a
        d = (1 + b * c) / a

        # Second chirp ϕ₂(k) = exp(i·π·d·k² / (N·b))
        phi2 = torch.exp(1j * math.pi * d * (idx ** 2) / (N * b))

        # Broadcast chirps over all batch dims, apply FFT along last axis.
        norm_mode = "ortho" if self.normalized else "backward"
        X = torch.fft.fft(x * phi1.to(x.dtype), dim=-1, norm=norm_mode)
        out = X * phi2.to(x.dtype)
        return out

    def inverse(self, X: Tensor) -> Tensor:
        """(Approximate) inverse transform. Uses the fact that the inverse of the LCT is the LCT with the inverse parameters."""
        # Compute inverse matrix [d, -b; -c, a]
        X_inv = X.inverse()

        return self.forward(X_inv)

    # ------------------------------------------------------------------
    # Helper utilities (to be completed later)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_d(self) -> Tensor:  # noqa: D401
        """Compute d from symplectic condition ad - bc = 1."""
        return (1 + self.b * self.c) / self.a if self.a != 0 else torch.tensor(1.0)
