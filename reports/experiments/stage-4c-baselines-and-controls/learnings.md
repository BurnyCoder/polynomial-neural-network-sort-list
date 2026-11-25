# Stage 4C Learnings: Pure Polynomials vs. Non-Polynomial Neural Sorting

**Date:** 2026-11-25

1. **Unstructured Sorting Bottleneck Is Architectural, Not Polynomial-Specific:** Parameter-matched `ReLU-MLP` (`0.0631` MAE at $n=16$, `0.0%` Ex@0.05) suffers virtually identical degradation to unstructured `PiNet-NCPSkip` (`0.0624`) and `PolyMLP` (`0.0653`), proving the bottleneck stems from missing pairwise sorting symmetry rather than polynomial activations.
2. **Why 6-Parameter `PolySortNet` Beats 96.6k-Parameter `SetTransformer` OOD:** Attention-based `SetTransformer` relies on learned value embeddings and quantile queries that fail catastrophically when inputs are shifted by $+100$ (`MAE = 99.22`) or extended $16\times$ to $n=128$ (`MAE = 0.0531`), and softens close keys in softmax attention on $10^{-3}$ near-ties (`0.00757`). By contrast, `PolySortNet`'s odd CAS structure algebraically guarantees shift invariance and handles ties/length extrapolation cleanly.
3. **Exact Price of Polynomial Purity vs. `DiffSort`:** Dropping $\exp/\arctan$ costs $\sim 2.3\times$ ID MAE vs. monotonic `DiffSort-Cauchy` (`0.0072` vs `0.0031` at $n=8$) and introduces a finite scale stability boundary ($R_{\text{stable}} = 1.0\text{--}1.3$ vs $5.0$), while unlocking 100% pure-polynomial execution (FHE CKKS & arithmetic circuits).
