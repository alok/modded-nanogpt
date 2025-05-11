# AGENT PLAN – NeurIPS 2025 LCT-NanoGPT Project

_Last updated: 2025-05-11 19:03 EDT_

## 0. TL;DR
Write a polished NeurIPS 2025 extended abstract showcasing a Linear Canonical Transform (LCT) layer inside NanoGPT; ship reproducible code, benchmarks, and documentation.

---

## 1. Immediate Next Actions

| Priority | Task                                                                      | Owner | ETA        |
| -------- | ------------------------------------------------------------------------- | ----- | ---------- |
| 🔥        | Draft abstract outline (`paper/outline.md`)                               | agent | 2025-05-12 |
| 🔥        | Scaffold `torchlayers/lct.py` with forward & inverse                      | agent | 2025-05-13 |
| 🔥        | Add unit test `tests/test_lct.py` (FFT reduction)                         | agent | 2025-05-14 |
| 🆕        | Add special-case tests `tests/test_lct_special.py` (Laplace/Fresnel/etc.) | agent | 2025-05-14 |
| 🆕        | Flesh out reference matrices in `tests/test_lct_special.py`               | agent | 2025-05-15 |
| ⚠️        | Justfile recipe `just bench:lct`                                          | agent | 2025-05-15 |

---

## 2. Milestones

1. **LCT Minimum Viable Layer**  
   Code complete & passes unit tests.
2. **Integration**  
   Swap NanoGPT's `nn.Linear` with `LCTLayer`; training script runs.
3. **Benchmark**  
   Tokens/sec baseline vs LCT plotted; results in `img/`.
4. **Special-Case Compliance**  
   All special-case tests (unitarity, Laplace, fractional Fourier, Fresnel) pass.
5. **Paper**  
   Extended abstract PDF passes NeurIPS style checker.
6. **Submission Package**  
   Tagged release `v0.1.0` on GitHub with Zenodo DOI.

---

## 3. Backlog / Ideas

* Investigate letting `a,b,c` be constrained (e.g., symplectic param).
* Explore FP8 support via `nn.quantized.Linear` wrappers.
* Use `torch.vmap` to batch LCT across sequence dimension.
* Ensure conjugate 2π convention w/ `normalized=True` matches `np.fft` ortho mode.

---

## 4. Decisions & Rationale (Chronological)

| Date       | Decision                                | Why                                 |
| ---------- | --------------------------------------- | ----------------------------------- |
| 2025-05-11 | Use Cursor Rule `neurips_2025_plan.mdc` | Always remind AI of deliverables    |
| 2025-05-11 | Deprecate `.cursorrules`                | Proper location is `.cursor/rules/` |

---

## 5. Glossary

* **LCT** – Linear Canonical Transform, param \(a,b,c\) governing affine symplectic mapping.
* **FFT** – Fast Fourier Transform; recovered when \(a=0, b=1, c=0\).

---

## 6. Next Update Trigger
When any task completes, open this file, tick box ✅, append to *Decisions* if applicable, and bump timestamp at top.
