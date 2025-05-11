# Linear Canonical Transform – Mathematical Background (draft)

> *This document is a working draft.  It will be expanded in parallel with the
> implementation.*

The **Linear Canonical Transform (LCT)** is the one–parameter family of
unitary integral transforms whose kernel is parameterised by a symplectic
matrix $S = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ with $ad-bc = 1$:

$$
\mathcal{L}\_{\{a,b,c,d\}}[f](y) =
\begin{cases}
  \dfrac{1}{\sqrt{2\pi i b}} \displaystyle\int\_{-\infty}^{\infty}
    \exp\!\left[\dfrac{i}{2b}(a x^2 - 2 x y + d y^2)\right] f(x)\,dx, & b \neq 0, \\
  \sqrt{d}\;e^{\tfrac{i}{2} c d y^2} f(d y), & b = 0.
\end{cases}
$$

Special cases include:

| $(a,b,c,d)$                 | Transform               |
|-----------------------------|-------------------------|
| $(0,1,0,0)$                 | Discrete Fourier (DFT)  |
| $(\cos\alpha, \sin\alpha, -\sin\alpha, \cos\alpha)$ | Fractional Fourier |
| $(1,0,-1,1)$                | Laplace                 |
| $(0,1,-1,0)$                | Fresnel (paraxial)      |

In our implementation we store the three free parameters $(a,b,c)$ and recover
$d$ from $ad-bc=1$ at runtime.

## Chirp–FFT–Chirp decomposition

For $b \neq 0$ the LCT can be evaluated in **$\mathcal{O}(N\log N)$** via the
`chirp → FFT → chirp` strategy (Campos & Figueroa, 2021):

$$
\mathcal{L}\_{a,b,c,d}[f](y) = C\, e^{i\pi \tfrac{a}{b} y^2}
\, \mathcal{F}\!\Big[ e^{i\pi \tfrac{1}{b} x^2} f(x) \Big]\!(\tfrac{y}{b}),
$$

where $\mathcal{F}$ is the unitary FFT and $C = 1/\sqrt{|b|}$ (up to
normalisation conventions).  This is the algorithm we will implement in
`torchlayers.lct.LCTLayer`.

*Further sections forthcoming.*
