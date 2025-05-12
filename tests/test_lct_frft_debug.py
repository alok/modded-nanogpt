from __future__ import annotations

import math

import torch
import pytest

# Removed early import of linear_canonical_transform; will import lazily in test function

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _frft_params(theta: float):
    """Return (a,b,c,d) for an FrFT of angle *theta*.

    Continuous‐time canonical matrix:
        [[cos θ,  sin θ],
         [−sin θ, cos θ]]
    """

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return cos_t, sin_t, -sin_t, cos_t


@pytest.mark.parametrize("N", [8, 16])
def test_two_frft_45_equals_fft(N: int) -> None:
    """Two successive 45° FrFTs should equal a 90° FrFT ≡ FFT.

    This is a *focused* regression for the composition bug observed in the
    generic property test.  A correct implementation should differ from the
    torch.fft oracle by at most a *global* phase (constant across indices).
    The test normalises that phase away and checks elementwise equality.
    """

    torch.manual_seed(0)
    x = torch.randn(N, dtype=torch.complex64)

    from torchlayers.functional.lct import linear_canonical_transform

    # Parameters for a 45° fractional Fourier transform (θ = π/4)
    a, b, c, d = _frft_params(math.pi / 4)

    # Sequential application: FrFT₍π/4₎ ∘ FrFT₍π/4₎
    y_seq = linear_canonical_transform(
        linear_canonical_transform(
            x,
            a=a,
            b=b,
            c=c,
            d=d,
            normalized=True,
            centered=True,
        ),
        a=a,
        b=b,
        c=c,
        d=d,
        normalized=True,
        centered=True,
    )

    # Oracle: unitary FFT equals a 90° FrFT
    y_fft = torch.fft.fft(x, norm="ortho")

    # Remove potential *global* phase & amplitude by normalising with one entry
    # (choose the first non-zero magnitude component to avoid division by 0).
    idx_ref = int(torch.argmax(torch.abs(y_fft)))
    phase = y_seq[idx_ref] / y_fft[idx_ref]

    assert phase.abs() > 1e-6, "Reference entry has near-zero magnitude."

    y_seq_norm = y_seq / phase

    assert torch.allclose(
        y_seq_norm,
        y_fft,
        atol=1e-4,
        rtol=0.0,
    ), "Composition of two 45° FrFTs should equal FFT up to global phase." 