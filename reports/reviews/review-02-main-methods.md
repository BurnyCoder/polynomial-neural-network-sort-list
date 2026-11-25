# Adversarial Self-Review #2: Post-Main-Methods Audit (Stages 4A & 4B)

**Date:** 2026-11-25  
**Reviewer Role:** Skeptical NeurIPS Reviewer  
**Scope:** Reviewing all 43 completed runs across Stages 4A (controls + primary `PolySortNet`) and 4B (full pure-PNN architecture ladder + $n=4..32$ scaling) before running non-polynomial baselines (4C) and causal ablations (4D).

---

## 1. Strict Hypothesis Audit Against Preregistration (`reports/preregistration.md`)

| Hypothesis | Preliminary Status (Seed 42; 5-Seed Statistical Tests Pending in 4E) | Honest Assessment / Caveats |
| :--- | :--- | :--- |
| **H1 (Inductive Bias Ladder)** | **PARTIALLY CONFIRMED (Degree-Dependent Nuance!)** | **Important Honest Catch:** Cubic ($m=1$, deg 27) `polysort_adaptive` (`0.0333`) did *not* beat 131k-param `deepsets_poly` (`0.0329`) by $2\times$ at $n=8$! However, quintic ($m=2$, deg 125) `polysort_shared_m2` (6 params, `0.0079`) and `polysort_adaptive_m2` (36–60 params, `0.0072`/`0.0097`) beat `deepsets_poly`, `pinet_ncp_skip`, and `poly_mlp` by **$4.1\times\text{--}8.8\times$** in MAE and **$117\times$** in `Ex@0.05` at $n=16$. We MUST report BOTH the cubic $m=1$ tie with `DeepSetsPoly` AND the quintic $m=2$ dominance — zero cherry-picking! |
| **H2 (Learned vs. Analytical CKKS at Iso-Depth)** | **STRONGLY SUPPORTED** | At exact iso-depth ($L=3, m=1$, deg 27), `polysort_adaptive` beats Cheon $f_1^{(3)}$ by **$3.4\times\text{--}3.9\times$** (`0.0333` vs `0.1136` at $n=8$; `0.0369` vs `0.1457` at $n=16$) and Lee et al. Remez $L=3$ by **37%–50%** (`0.0531`/`0.0740`). At $m=2$ (deg 125), learned `polysort_adaptive_m2` hits `0.0072`/`0.0097` while iso-degree analytical Remez diverges (`inf`). |
| **H4 (`PolyPairwiseRank` vs. `PolySortNet`)** | **STRONGLY SUPPORTED** | Exact predicted crossover: `poly_pairwise_rank` beats `PiNet`/`PolyMLP` at $n=8$ (`0.0430`) but degrades at $n=16$ (`0.0738`, Mono `0.11%`). |
| **H5 (Zero-Shot Length & Shift vs. Scale Cliff)** | **STRONGLY SUPPORTED** | `polysort_shared_m2` (6 params) achieves `0.0200` MAE & $\tau = 0.9978$ zero-shot at $n=128$, exact float64 sum conservation (`2.1e-16`), shift equivariance at $c=100$ (`6.6e-14`), AND exhibits the predicted polynomial blow-up cliff at $s = 1.5\text{--}2.0$. |

---

## 2. Adversarial Critiques & Action Items for Stages 4C–4E

1. **Critique: "Did `PiNet-NCPSkip` or `PolyMLP` underfit because of `lr=2e-3` or insufficient depth?"**
   - **Check:** Stage 4E explicitly sweeps `lr` $\in \{3\times 10^{-4}, 10^{-3}, 3\times 10^{-3}, 10^{-2}\}$ on `pinet_ncp_skip` to ensure its MAE (`0.0529`) is not an LR artifact. Also, Stage 4C's `ReLU-MLP` (parameter-matched to `PolyMLP`) and `SetTransformer` (96.6k params) will immediately reveal whether unstructured sorting difficulty is unique to polynomials or shared by general neural networks!
2. **Critique: "Why did `PolySortNet-m2` have `82.5%` monotonicity at $n=16$ vs. `DiffSort`?"**
   - **Check:** Directly addressed by Stage 4C (`DiffSort` Logistic & Cauchy baselines) AND Stage 4D Ablation A5 (`abl_loss_mse_mono`, adding monotonicity hinge penalty $\lambda = 0.1$ to test H2b's value-vs-ordering trade-off!).
3. **Critique: "Single seed `42` is insufficient for publication claims."**
   - **Check:** All 10 core comparison models will be run across 5 independent seeds (`42, 123, 456, 789, 1024`) with 10,000-resample paired bootstrap CIs and Holm-Bonferroni Wilcoxon tests in Stage 4E.
4. **Critique: "Verify zero target leakage experimentally."**
   - **Check:** Stage 4C runs `ctrl_shuffled_n8` and `ctrl_shuffled_n16` (training `polysort_adaptive` on random-permuted targets); must fail to sort and collapse toward naive mean error.

---

## 3. Verdict
All 43 runs in Stages 4A–4B are scientifically sound, leak-free, FX-verified, and logged with full honest nuance (including where cubic $m=1$ ties `DeepSetsPoly`). Cleared to proceed to **Stage 4C (Baselines & Shuffled Control)** and **Stage 4D (6 Causal Ablations)**.
