# NeurIPS 2025 Paper Outline: Adaptive LCT Layers

## 0. Abstract (sections/00_abstract.tex)
- TODO: Concise summary (contributions, methods, key results, significance).

## 1. Introduction (sections/01_introduction.tex)
- Motivation: Efficiency and adaptivity in Transformer layers.
- Problem: Standard linear/Fourier layers are fixed; can we learn better transforms?
- Proposal: Adaptive Linear Canonical Transform (LCT) layers.
- Contributions:
  - `LCTLayer` PyTorch module with learnable parameters.
  - Integration into NanoGPT.
  - Benchmarks (speed, perplexity).
- Roadmap of the paper.

## 2. Related Work (sections/02_related_work.tex)
- LCTs in signal processing and optics.
- Fourier transforms and variants (FrFT, FFT) in deep learning (e.g., FNet).
- Efficient Transformer architectures (alternatives to dense attention/FFN).
- Adaptive/dynamic neural network components.

## 3. Method: The LCT Layer (sections/03_method.tex)
- **Linear Canonical Transform (LCT) Background**
  - Definition, ABCD matrix, symplectic property.
- **Discrete LCT Implementation (`LCTLayer`)**
  - `LCTLayer(nn.Module)` for 1D signals.
  - Learnable \(a, b, c\); \(d\) derived.
  - Forward pass: `linear_canonical_transform` details (chirp-FFT-chirp, scaling, dense kernel).
  - Analytical inverse.
  - Properties: `centered`, `normalized`, JIT, GPU, bf16/fp8.
- **Integration into NanoGPT-style Models**
  - Replacement strategy for `nn.Linear`.
  - Parameterization choices.

## 4. Experiments (sections/04_experiments.tex)
- **Setup**
  - Dataset (TinyShakespeare / FineWeb sample).
  - Baseline NanoGPT architecture.
  - LCT-NanoGPT: modified architecture, LCT layer placement, initialization.
  - Training: optimizer, LR, batch, steps, hardware (Modal/H100).
  - Metrics: PPL, loss, tokens/sec, wall time.
- **Baseline Comparison**
  - Training curves (loss, PPL).
  - Final metrics table.
- **Ablation Studies (Optional)**
  - LCT initialization.
  - LCT placement.

## 5. Results (sections/05_results.tex)
- Presentation of Table \ref{tab:main_results}.
- Presentation of Figure \ref{fig:speed_accuracy}.
- Discussion of quantitative and qualitative results.
- Summary of gains.

## 6. Discussion (sections/06_discussion.tex)
- Interpretation of experimental findings.
- Significance of adaptive LCTs.
- Limitations of the current work (e.g., 1D LCTs, scope of experiments).
- Potential failure modes or negative results if observed.

## 7. Conclusion (sections/07_conclusion.tex)
- Recap of contributions and key findings.
- Future work:
  - 2D LCTs for vision.
  - Broader applications of adaptive transforms.
  - Deeper theoretical understanding of learned LCT parameters.

## 8. Reproducibility and Broader Impact (sections/08_reproducibility.tex)
- **Reproducibility Statement**
  - Code, data, environment details.
- **Broader Impact Statement**
  - Positive impacts (efficiency, new modeling capabilities).
  - Potential negative impacts (general LLM concerns).

## 9. Checklist (sections/99_checklist.tex)
- NeurIPS checklist completion.

## References (references.bib)
- TODO: Add citations as we write.

## Appendix (in main.tex or separate file)
- Detailed proofs (if any).
- Extended experimental results or ablations.
- Further implementation details. 