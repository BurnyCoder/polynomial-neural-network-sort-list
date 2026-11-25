# Planned Experimental Run Matrix

**Author:** Libor "Burny" Burian  
**Date:** 2026-11-25  
**Companion Document:** [preregistration.md](preregistration.md)

This document enumerates every planned training, control, baseline, ablation, and sensitivity run, explicitly specifying how each run differs in architecture, polynomial degree, multiplicative depth, wiring, loss, optimizer, data, and seed.

---

## 1. Shared Default Training Protocol

- **Training Set:** $200{,}000$ lists from $\mathrm{Uniform}[-1, 1]^n$ (seed-isolated `train` stream).
- **Validation / Test Sets:** $10{,}000$ / $10{,}000$ lists (seed-isolated `val`, `test` streams).
- **Default Optimization:** `AdamW`, batch size `512`, `60` epochs (`~23,400` optimizer steps), cosine annealing LR schedule with 5% linear warmup, gradient norm clipping `1.0`.
  - Default LR: `3e-3` for compact structured polynomial networks (`polysort_*`, `poly_pairwise_rank`), `1e-3` for deep dense networks (`pinet_ncp_skip`, `poly_mlp`, `deepsets_poly`, `relu_mlp`, `set_transformer`), `1e-2` for `diffsort` learnable steepness / linear control. (Verified in Stage 4E LR sensitivity sweep.)
- **Default Loss:** Value MSE $\mathcal{L}_{\text{MSE}}(\hat{y}, y) = \frac{1}{n}\|\hat{y} - y\|_2^2$ (loss ablations tested in Stage 4D).
- **Precision:** Trained in `float32` (with `float64` training check on deep $L=4$ composite ablation and `float64` verification at evaluation time).
- **Checkpointing:** Atomic checkpoint after every epoch (saving model, optimizer, scheduler, RNG states, best validation checkpoint) for full restart resumability.

---

## 2. Stage 4A — Zero-Training Controls, Analytical Baselines & Primary Main Models (Single Seed = 42)

| Run ID | Model Family | Pure Poly? | Exact Architecture & Diff | Comp. Deg ($D_s$) | Seq. Mult. Depth | $n$ | Training? |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| `ctrl_identity_n8/n16` | `IdentityControl` | Yes | $\hat{y} = x$ (no-op lower bound) | 1 | 0 | 8, 16 | None |
| `ctrl_linear_n8/n16` | `LinearControl` | Yes | Exact closed-form Ridge/OLS $\hat{y} = Wx + b$ fit on `train` | 1 | 0 | 8, 16 | Closed-form |
| `ctrl_rand_polysort_n8/n16` | `RandomInitControl` | Yes | Untrained random-init `PolySortNet-Shared` ($L=3, d=3$) | 27 | $6\times 3$ / $10\times 3$ | 8, 16 | None |
| `ctrl_rand_pinet_n8/n16` | `RandomInitControl` | Yes | Untrained random-init `PiNet-NCPSkip` | 6 | 5 | 8, 16 | None |
| `ana_cheon_f1_L3_n8/n16` | `AnalyticalCheonSort` | Yes | Bitonic wiring + Cheon $f_1^{(3)}$ ($L=3, d=3$) | 27 | $3\ell$ | 8, 16 | None |
| `ana_cheon_f2_L2_n8/n16` | `AnalyticalCheonSort` | Yes | Bitonic wiring + Cheon $f_2^{(2)}$ ($L=2, d=5$) | 25 | $2\ell$ | 8, 16 | None |
| `ana_cheon_fg_n8/n16` | `AnalyticalCheonFGSort` | Yes | Bitonic wiring + Cheon/Hong/Mazzone mixed $f_3 \circ g_3$ ($L=2, d=7$) | 49 | $2\ell$ | 8, 16 | None |
| `ana_remez_L3_n8/n16` | `AnalyticalRemezSort` | Yes | Bitonic wiring + Lee et al. (2022) HiGHS LP minimax odd cubic ($L=3, d=3, \delta=0.10$) | 27 | $3\ell$ | 8, 16 | None (LP fit) |
| `ana_remez_L3_m2_n8/n16` | `AnalyticalRemezSort` | Yes | Bitonic wiring + Lee et al. (2022) HiGHS LP minimax odd quintic ($L=3, d=5, \delta=0.03$) | 125 | $3\ell$ | 8, 16 | None (LP fit) |
| **`main_polysort_shared_n8/n16`** | `PolySortNet` (Shared) | Yes | Bitonic wiring + **shared** fixed-point odd Chebyshev CAS ($L=3, m=1 \Rightarrow d=3, D_s=27$, 3 params) | 27 | $3\ell$ | 8, 16 | AdamW, 60 ep |
| **`main_polysort_shared_m2_n8/n16`** | `PolySortNet` (Shared-Deg5) | Yes | Bitonic wiring + **shared** fixed-point odd Chebyshev CAS ($L=3, m=2 \Rightarrow d=5, D_s=125$, 6 params) | 125 | $3\ell$ | 8, 16 | AdamW, 60 ep |
| **`main_polysort_adaptive_n8/n16`** | `PolySortNet` (Stage-Adaptive) | Yes | Bitonic wiring + **per-layer** fixed-point odd Chebyshev CAS ($L=3, m=1 \Rightarrow d=3$, $3\ell$ params: 18 for $n=8$, 30 for $n=16$) | 27 | $3\ell$ | 8, 16 | AdamW, 60 ep |
| **`main_polysort_adaptive_m2_n8/n16`** | `PolySortNet` (Stage-Adaptive-Deg5) | Yes | Bitonic wiring + **per-layer** fixed-point odd Chebyshev CAS ($L=3, m=2 \Rightarrow d=5$, $6\ell$ params) | 125 | $3\ell$ | 8, 16 | AdamW, 60 ep |

*(Note: $\ell = 6$ for Bitonic $n=8$; $\ell = 10$ for Bitonic $n=16$; $\ell = 15$ for $n=32$; $\ell = 21$ for $n=64$; $\ell = 28$ for $n=128$.)*

---

## 3. Stage 4B — Remaining Main Pure-PNN Architectures & Scaling Study (Single Seed = 42)

| Run ID | Model Family | Pure Poly? | Exact Architecture & Diff | Params ($n=8/16$) | $n$ |
| :--- | :--- | :---: | :--- | :---: | :---: |
| `main_poly_pairwise_n8/n16` | `PolyPairwiseRank` | Yes | Mazzone-inspired all-to-all $O(1)$ comp depth: learned odd Chebyshev $s_{\text{cmp}}$ ($L=3, m=2$) + exact tie-correction $E=4C(1-C)$ + learned indicator $s_{\text{ind}}$ ($L=3, m=2$) | 12 / 12 | 8, 16 |
| `main_polysort_learnedwire_n8/n16` | `PolySortNetLearnedWire` | Yes | $\ell$ stages of learned residual linear routing matrices $R_t \in \mathbb{R}^{n\times n}$ interleaved with fixed-point Chebyshev CAS gates ($L=3, m=1$) | ~402 / ~2,590 | 8, 16 |
| `main_deepsets_poly_n8/n16` | `DeepSetsPoly` | Yes | Permutation-invariant $\rho_{\text{poly}}(\frac{1}{n}\sum_i \phi_{\text{poly}}(x_i))$, $\phi$: deg-6 Chebyshev features + linear ($H=64 \ge 2n$), $\rho$: 3-block ProdPoly NCP-Skip | ~35k / ~36k | 8, 16 |
| `main_pinet_ncpskip_n8/n16` | `PiNetSorter` | Yes | Chrysos et al. (2021) ProdPoly NCP-Skip (2 stacked deg-3 NCP-Skip blocks, hidden width 128, zero BN/tanh, exact Eq. 10) | ~52k / ~56k | 8, 16 |
| `main_poly_mlp_n8/n16` | `PolyMLP` | Yes | 3-hidden-layer dense MLP (width 128) with per-neuron learnable deg-3 Chebyshev polynomial activations (Kileel et al., 2019) | ~36k / ~38k | 8, 16 |
| `scale_polysort_n4/n32` | `PolySortNet` (Shared & Adaptive) | Yes | Scaling study across $n=4$ ($\ell=3$) and $n=32$ ($\ell=15$) + zero-shot eval of `n8`-trained `polysort_shared` on $n \in \{4, 8, 16, 32, 64, 128\}$ | 3–90 | 4, 32 (+ zero-shot 64, 128) |
| `scale_unstruct_n4` | `PiNetSorter`, `PolyMLP`, `DeepSetsPoly` | Yes | Small-$n$ regime ($n=4$) to verify whether unstructured PNNs succeed on short lists and quantify how rapidly they degrade with $n$ | ~34k–50k | 4 |

---

## 4. Stage 4C — Trained Control & Non-Polynomial Baselines (Single Seed = 42)

| Run ID | Model Family | Pure Poly? | Exact Architecture & Diff | $n$ |
| :--- | :--- | :---: | :--- | :---: |
| `ctrl_shuffled_n8/n16` | `ShuffledTargetControl` | Yes | `PolySortNet-Adaptive` trained on randomly permuted targets (leakage check) | 8, 16 |
| `base_diffsort_logistic_n8/n16` | `DiffSortBaseline` (Logistic) | **No** | Petersen et al. (2021) Bitonic + logistic sigmoid $\sigma(\beta(b-a))$, learned log-steepness $\beta$ | 8, 16 |
| `base_diffsort_cauchy_n8/n16` | `DiffSortBaseline` (Cauchy) | **No** | Petersen et al. (2022) Bitonic + monotonic Cauchy CDF $\frac{1}{\pi}\arctan(\beta(b-a)) + \frac{1}{2}$, learned $\beta$ | 8, 16 |
| `base_relu_mlp_n8/n16` | `ReLUMLPBaseline` | **No** | 3-layer GELU/ReLU MLP (width 128, parameter-matched to `PolyMLP` / `PiNet`) | 8, 16 |
| `base_set_transformer_n8/n16` | `SetTransformerBaseline` | **No** | 2-layer multi-head SAB + PMA-style slot queries (Lee et al., 2019; width 64, 4 heads, ~40k params) | 8, 16 |

---

## 5. Stage 4D — Causal Ablation Suite ($n = 8$ & $n = 16$, Seed = 42)

| Ablation Family | Run IDs | Controlled Variable Changed (All Else Fixed to `PolySortNet-Adaptive` Bitonic) |
| :--- | :--- | :--- |
| **A1. Parameterization** | `abl_param_fp_cheby` (ref), `abl_param_unconst_odd`, `abl_param_general_cheby`, `abl_param_raw_monomial` | Fixed-point odd Chebyshev vs. unconstrained odd Chebyshev vs. general (even+odd) Chebyshev vs. raw monomials |
| **A2. Iso-Degree Depth** | `abl_depth_L1_d9` (flat deg 9), `abl_depth_L2_d3` (comp deg 9), `abl_depth_L3_d3` (deg 27), `abl_depth_L4_d3` (deg 81) | Tests Bernstein vs. Cheon depth separation at exact iso-degree $D=9$ and depth scaling $L \in \{1, 2, 3, 4\}$ |
| **A3. Initialization** | `abl_init_identity` ($c=0$), `abl_init_random` ($\mathcal{N}(0, 0.1^2)$), `abl_init_cheon` (Cheon $f_1$ warm-start) | Tests optimization landscape sensitivity and convergence speed |
| **A4. Wiring Topology** | `abl_wire_bitonic` ($\ell=6/10$), `abl_wire_oddeven_merge` (Batcher odd-even merge), `abl_wire_oddeven_trans` ($\ell=8/16$), `abl_wire_random_fixed` (**negative control**) | Isolates causal contribution of valid sorting-network wiring vs. random pairwise swaps |
| **A5. Objective Loss** | `abl_loss_mse` (ref), `abl_loss_huber`, `abl_loss_mse_mono` (MSE + $0.1 \cdot \mathrm{ReLU}(\hat{y}_k - \hat{y}_{k+1})$), `abl_loss_mse_mono_w1` | Tests H2b trade-off between Value MAE and Adversarial Near-Tie Ordering |
| **A6. Scale Boundary (Non-Poly)** | `abl_scale_pure` (ref) vs. `abl_scale_norm_maxabs` (divides by $\max_i |x_i| + 10^{-8}$ before/after — **explicitly non-polynomial**) | Quantifies exact scale-extrapolation trade-off of strict polynomial purity |

---

## 6. Stage 4E — Multi-Seed Statistical Rigor & Hyperparameter Sensitivity

- **Multi-Seed Suite (5 Seeds: `42, 123, 456, 789, 1024`) on $n = 8$ and $n = 16$:**
  - `polysort_shared`, `polysort_adaptive`, `polysort_adaptive_m2`, `poly_pairwise_rank`, `deepsets_poly`, `pinet_ncp_skip`, `poly_mlp`, `diffsort_cauchy`, `relu_mlp`, `set_transformer` (10 models $\times$ 2 lengths $\times$ 5 seeds = 100 runs; ~45–75 min total on RTX 5070).
- **Hyperparameter Sensitivity Sweep ($n = 8$, `polysort_adaptive` & `pinet_ncp_skip`):**
  - Learning rate $\in \{3\times 10^{-4}, 10^{-3}, 3\times 10^{-3}, 10^{-2}\}$.
  - Weight decay $\in \{0, 10^{-4}, 10^{-2}\}$.
  - Batch size $\in \{128, 512, 2048\}$.
  - Precision check: `float32` vs. `float64` training on `polysort_adaptive` ($L=3$ and $L=4$).
