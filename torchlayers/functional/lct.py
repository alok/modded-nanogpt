"""Low-level helpers for the Linear Canonical Transform.

This module is intentionally *minimal* at the moment.  Only API stubs are
defined so that other parts of the codebase can import these symbols without
failing.  Full numerical kernels will be added in forthcoming commits.
"""

from __future__ import annotations

import torch

# Type alias
Tensor = torch.Tensor

__all__ = [
    "chirp_multiply",
    "symplectic_d",
]


def chirp_multiply(x: Tensor, coef: Tensor) -> Tensor:  # noqa: D401
    """Multiply input *x* element-wise by a quadratic phase (chirp).

    Placeholder – raises until implemented.
    """

    raise NotImplementedError


def symplectic_d(a: Tensor, b: Tensor, c: Tensor) -> Tensor:  # noqa: D401
    """Compute *d* so that the matrix [[a, b], [c, d]] has determinant 1."""

    return (1 + b * c) / a
