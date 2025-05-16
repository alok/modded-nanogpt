## Rapid-Tuning Use Cases for the Linear Canonical Transform (LCT) Layer

Each scenario assumes the `torchlayers.LCTLayer` has been swapped in for the projection or activation function under study.  All three can be prototyped in a _single afternoon_ on a single GPU.

1. **Fresnel-Vision Attention**  
   *Idea*: Replace the softmax in self-attention with an LCT parameterised as the Fresnel transform \((a=0,\,b=1,\,c=1)\).  The quadratic phase term favours nearer tokens, providing an inductive bias for local syntax without hard windows.
   *Rapid tuning*: Only the scalar \(c\) is learnable; sweep over \(c \in [0,2] \) with a 1-liner in the config.
   *Hypothesis*: Improves convergence speed on long-sequence language modelling.

2. **Beam-Splash Activation for Diffusion Models**  
   *Idea*: Insert an LCT layer with small positive \(b\) (fractional Fourier regime) after each UNet residual block.  The transform spreads frequency content (`splash'`) encouraging smoother denoising trajectories.
   *Rapid tuning*: Grid-search \(b\in\{0.2,0.4,0.6\}\) while freezing \(a=c=0\).
   *Metric*: FID after 10k training steps on CIFAR-10.

3. **Mirror-Symmetry Mixer**  
   *Idea*: For image encoders, apply an LCT with \(a=1,\,b=0,\,c=-1\) (optical mirror) to channel activations before the classification head.  Allows the network to explicitly model reflective symmetries.
   *Rapid tuning*: Finetune only \(c\) around \(-1\) using a cosine schedule.
   *Dataset*: STL-10 — test whether data-efficient representation improves by ≥2 pp top-1. 