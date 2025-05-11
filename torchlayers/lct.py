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

    def __init__(self, *, a: float = 0.0, b: float = 1.0, c: float = 0.0, normalized: bool = True) -> None:  # noqa: D401,E501
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a)))
        self.b = nn.Parameter(torch.tensor(float(b)))
        self.c = nn.Parameter(torch.tensor(float(c)))
        self.normalized = normalized

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        """Apply the LCT.

        Currently unimplemented; raises to remind developers to finish the
        kernel.  Tests are marked *xfail* until then.
        """

        raise NotImplementedError("LCT forward kernel not implemented yet.")

    def inverse(self, X: Tensor) -> Tensor:
        """Approximate inverse transform (placeholder)."""

        raise NotImplementedError("LCT inverse kernel not implemented yet.")

    # ------------------------------------------------------------------
    # Helper utilities (to be completed later)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_d(self) -> Tensor:  # noqa: D401
        """Compute d from symplectic condition ad - bc = 1."""
        return (1 + self.b * self.c) / self.a if self.a != 0 else torch.tensor(1.0)
