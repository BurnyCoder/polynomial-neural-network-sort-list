# Stage 4B Learnings: Inductive Bias Ladder, Monotonicity Gaming, and Crossover Laws

**Date:** 2026-11-25

1. **Honest Nuance on H1 — Symmetric Polynomials (`DeepSetsPoly`) Are Surprisingly Strong In-Distribution at Small $n$, But Brittle OOD:**
   - Enforcing permutation invariance via Chebyshev power sums ($\phi(x_i)$ mean-pooled with latent width $\ge 4n$, respecting Wagstaff et al. 2019/2022) cuts error by 35–50% relative to unstructured `PiNet-NCPSkip` and `PolyMLP`, reaching `0.0121` MAE at $n=4$ and tying cubic ($m=1$) `polysort_adaptive` at $n=8$ (`0.0329` vs `0.0333`).
   - **However:** (a) Quintic ($m=2$) `PolySortNet` crushes `DeepSetsPoly` by **$4.1\times\text{--}4.5\times$** in MAE and **$117\times$** in Exact-Sort@0.05 at $n=16$ (`82.0%` vs `0.7%`) using $2{,}200\times$ fewer parameters; (b) `DeepSetsPoly` explodes to `inf` under $+10$ shift OOD and degrades $2.5\times$ on 50% exact duplicates, whereas `PolySortNet` is shift-invariant to machine precision and *improves* on duplicates!
2. **H4 Nailed Exactly — Constant-Depth `PolyPairwiseRank` (Mazzone-Inspired) Crossover:**
   - At $n=8$, 12-parameter `PolyPairwiseRank` (comparison depth 2) beats 201k-param `PiNet` and 37k-param `PolyMLP` (`0.0430` vs `0.0529`/`0.0641`).
   - At $n=16$, its rank indicator polynomial must distinguish rank spacing $1/(2n) = 0.03125$, causing MAE to jump to `0.0738` and monotonicity to collapse to `0.11%`. Moreover, because $\sum_j W_{jk} \neq 1$ identically, `PolyPairwiseRank` loses exact shift equivariance (`Shift +10 MAE = 1.02` vs `PolySortNet`'s `0.0073`).
3. **Why Raw Monotonicity Can Mislead Without Value-Matched Exact Sort Rate (`Ex@0.05`):**
   - `LinearControl` (`99.7%`), `PiNet` (`98.6%`), and `PolyMLP` (`98.5%`) exhibit higher raw monotonicity than `PolySortNet` (`82.5%`) at $n=16$ despite $6\times\text{--}10\times$ worse MAE! The reason: MSE regression in unstructured nets shrinks predictions toward the widely spaced expected order statistics $\mathbb{E}[x_{(k)}]$ (distorting the multiset values, as confirmed by high $W_1 = 0.062\text{--}0.099$ and `0.0%` `Ex@0.05`), whereas `PolySortNet` preserves exact input values (`W1 = 0.0097`, `Ex@0.05 = 82.0%`).
4. **Monotonically Growing Scaling Law for Stage-Adaptive Specialization:**
   - Holding comparator degree fixed at 27 ($m=1$), stage-adaptive specialization improves over shared-gate `PolySortNet` by `-6.2%` ($n=4$) $\to$ `-11.9%` ($n=8$) $\to$ `-23.7%` ($n=16$) $\to$ **`-31.5%` ($n=32$)**, because deeper Bitonic networks experience a wider ratio between initial random gaps ($O(1)$) and final merge gaps ($O(1/n)$).
5. **Unconstrained Continuous Wiring Fails to Discover Batcher Topology:**
   - `PolySortNetLearnedWire` (orthogonal-initialized linear routing interleaved with poly CAS) beats `LinearControl` by 24–28% (`0.0967` at $n=8$, `0.0753` at $n=16$), proving it learns *some* nonlinear sorting, yet trails fixed Bitonic `PolySortNet` by $7.8\times\text{--}13.3\times$ and loses sum conservation and shift equivariance.
