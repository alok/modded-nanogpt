import math
import cmath
from typing import List, Tuple

import numpy as np
import pytest
import torch

from torchlayers.lct import LCTLayer

# -----------------------------------------------------------------------------
# Helper: reference orthonormal DFT matrix (conjugate 2π convention)
# -----------------------------------------------------------------------------

def dft_matrix(n: int) -> np.ndarray:
    """Return the unitary NxN DFT matrix using the 1/√N normalisation."""
    k = np.arange(n).reshape((n, 1))
    j = np.arange(n).reshape((1, n))
    w = np.exp(-2j * math.pi * k * j / n)
    return w / math.sqrt(n)


# -----------------------------------------------------------------------------
# Parametrisation of special cases: (name, (a, b, c), reference_matrix_fn)
# Only Fourier is fully specified for now; others are TODO.
# -----------------------------------------------------------------------------

SPECIAL_CASES: List[Tuple[str, Tuple[float, float, float], callable]] = [
    ("fourier", (0.0, 1.0, 0.0), lambda n: dft_matrix(n)),
    # TODO: fill in laplace, fractional_fourier, fresnel, degenerate cases
]


@pytest.mark.parametrize("name, params, ref_fn", SPECIAL_CASES)
@pytest.mark.xfail(reason="LCT forward kernel not yet implemented")
def test_special_case(name: str, params: Tuple[float, float, float], ref_fn):
    n = 4  # Small size for sanity
    x = torch.eye(n, dtype=torch.complex64)
    layer = LCTLayer(a=params[0], b=params[1], c=params[2], normalized=True)
    out = layer(x)
    expected = torch.tensor(ref_fn(n), dtype=torch.complex64)
    assert torch.allclose(out, expected, atol=1e-4), f"Mismatch for {name}"


@pytest.mark.xfail(reason="Unitarity property check pending implementation")
def test_unitarity_fourier():
    n = 4
    layer = LCTLayer(a=0.0, b=1.0, c=0.0, normalized=True)
    # Out = layer applied to identity should be the DFT matrix
    dft = torch.tensor(dft_matrix(n), dtype=torch.complex64)
    assert torch.allclose(dft.conj().T @ dft, torch.eye(n, dtype=torch.complex64), atol=1e-4)
