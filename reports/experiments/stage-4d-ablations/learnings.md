# Learnings from Stage 4D: Causal Ablations (A1–A6)

**Author:** Libor "Burny" Burian  
**Date:** 2026-11-25  

1. **Why Unconstrained Polynomials Beat Fixed-Point Polynomials In-Domain (and Why That's Dangerous Out-of-Domain):**
   - *Initial expectation (H6 Part 1):* Removing the $h_k(1) = 1$ fixed-point anchor would hurt accuracy or destabilize training.
   - *Actual causal observation:* Unconstrained odd Chebyshev, raw monomial, and general Chebyshev all converged cleanly (0% divergence) and reduced ID MAE by **26–30%**!
   - *Mechanistic root cause (verified via float64 transfer inspection):* Late sorting stages receive pre-sorted inputs whose pairwise half-differences rarely exceed $|\Delta| \approx 0.4$. Without the $h_k(1)=1$ anchor, gradient descent exploits this by shrinking Stage 5's effective operating range, pushing origin slope $s'(0)$ from `9.58 -> 14.3` while letting $s_5(1.0)$ blow up to **`-444` to `-1038`**. Consequently, any input whose scale expands to $1.3\times$ explodes immediately (`R_stable: 1.3 -> 1.1`).
   - *Takeaway:* Fixed-point anchoring is not an accuracy maximizer on bounded $[-1,1]$ data — it is a **scale-stability regularizer** that enforces global boundedness over $[-1,1]$ across every stage!

2. **Iso-Degree vs. Iso-Parameter Depth Separation:**
   - *Initial expectation (H3 Part 1):* Composite degree 9 (`L=2, d=3`) would beat flat degree 9 (`L=1, d=9`).
   - *Actual causal observation:* Flat `L=1, d=9` (4 params/stage, MAE `0.0566 / 0.0674`) beat composite `L=2, d=3` (2 params/stage, MAE `0.0771 / 0.0831`) on ID MAE, **because** the 2D composite variety is a strict algebraic subset of the 4D flat degree-9 space.
   - *However*, flat degree 9 suffers severe **Runge-type boundary blow-up** (`s(1.1) = 1.92`, `R_stable = 0.00`), whereas matching parameter count (`4 params/stage`) with composite **`L=4, d=3` (degree 81)** yields a **$4.28\times$ MAE drop**, **`91.8%` / `61.4%` Ex@0.05**, AND expands stability to **`R_stable = 1.50`** (higher than BOTH `L=3, m=1` (`1.30`) and `L=3, m=2` (`1.00`))!
   - *Design lesson:* Going **deeper with low-degree cubics (`L=4, d=3`)** achieves a dramatically better accuracy–stability frontier (`R_stable = 1.50`, MAE `0.0132/0.0163`) than going **wider with higher-degree flat polynomials (`L=1, d=9`)** or even quintics (`L=3, m=2`, `R_stable = 1.00`).

3. **Hidden Symmetry Trade-Off in Non-Polynomial Max-Abs Normalization (A6):**
   - Wrapping `PolySortNet` in non-polynomial `x / max(|x|)` solves scale extrapolation (`R_stable = 5.0`), proving H5 causally, *but* unexpectedly degrades Shift $+100$ MAE by **$13\times$** (`0.0334 -> 0.4553`), because uncentered max-abs scaling shrinks all pairwise differences by $101\times$. Pure difference-based `PolySortNet` preserves exact shift equivariance (`7.75e-14` in float64) automatically; only a *centered* non-polynomial normalizer `(x - mean(x)) / max(|x - mean(x)|)` could achieve both simultaneously.

4. **Reversed-Input Asymmetry in Odd-Even Transposition Wiring (A4):**
   - Habermann Odd-Even Transposition delivers the highest monotonicity (`99.30%` at $n=8$, `96.77%` at $n=16$) because adjacent final passes polish remaining inversions, *but* incurs a **$2.74\times$ error penalty on reversed inputs vs. sorted inputs** at $n=16$ (`0.1349` vs `0.0493`), because elements must traverse all $n$ sequential bubble steps — whereas Bitonic and Batcher Odd-Even Merge are graph-symmetric under reversal (`EdgeS == EdgeR` identically).
