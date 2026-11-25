# Scientific Preregistration: Strictly Pure Polynomial Neural Networks for List Sorting

**Author:** Libor "Burny" Burian  
**Preregistration Date (Pre-Results):** 2026-11-25  
**Status:** LOCKED BEFORE ANY TRAINING OR EVALUATION RESULTS EXIST  
**Repository:** `BurnyCoder/polynomial-neural-network-sort-list`

> **Scientific Integrity Commitment:** This document records all hypotheses, primary metrics, experimental controls, statistical tests, and explicit falsification criteria **prior to executing any model training or evaluation** (following Lones, [arXiv:2108.02497](https://arxiv.org/abs/2108.02497); Lipton & Steinhardt, [arXiv:1807.03341](https://arxiv.org/abs/1807.03341); and NeurIPS Paper Checklist standards). Any subsequent exploratory analyses will be explicitly tagged as **Post-Hoc** in Section 7.

---

## 1. Domain, Task, and Strict Pure-Polynomial Definition

- **Task:** Given $x = (x_1, \dots, x_n) \in [-1, 1]^n$, predict the non-decreasing sorted vector $y = \mathrm{sort}(x) = (x_{(1)}, \dots, x_{(n)})$ with $x_{(1)} \le \dots \le x_{(n)}$.
- **Strict Pure-Polynomial Invariant:** Every model classified as a **Pure PNN** must compute $f_\theta(x)$ using **exclusively** additions, subtractions, constant/learned affine maps, and multiplications during inference (no ReLU, GELU, sigmoid, tanh, softmax, division, square root, `abs`, `min`/`max`, `argsort`, or runtime normalization). Purity is audited mechanically via PyTorch FX graph tracing (`tests/test_purity.py`).

---

## 2. Pre-Specified Hypotheses & Explicit Falsification Criteria

### H1 — Architectural Inductive Bias Ladder (Primary Hypothesis)
- **Claim:** On in-distribution Uniform $[-1, 1]^n$ list sorting ($n \in \{8, 16\}$), strictly pure polynomial sorting networks (`polysort_shared`, `polysort_adaptive`) achieve at least **$5\times$ lower Value MAE** and significantly higher **Exact Monotonicity Rate** and **Kendall's $\tau$** than unstructured pure PNNs (`pinet_ncp_skip`, `poly_mlp`) and permutation-invariant power-sum PNNs (`deepsets_poly`), while using $\ge 10\times$ fewer learnable parameters.
- **Falsification Criterion:** H1 is **falsified** if `polysort_shared` or `polysort_adaptive` fails to beat all three (`pinet_ncp_skip`, `poly_mlp`, `deepsets_poly`) on $n=8$ and $n=16$ Value MAE by at least $2\times$ with Holm-Bonferroni corrected $p < 0.01$ (paired Wilcoxon signed-rank test on $N_{\text{test}}=10{,}000$ lists, averaged across 5 seeds).

### H2 — Learned Composite PNNs vs. Zero-Training Analytical CKKS Minimax at Iso-Depth
- **Claim:** At the **exact same sorting-network wiring, composition depth $L$, and polynomial degree $d$** (Bitonic network, $L=3$ cubic layers, comparator degree $3^3 = 27$), gradient-trained `polysort_adaptive` (and `polysort_shared`) will achieve **strictly lower Value MAE** on Uniform $[-1, 1]^n$ ($n=8, 16$) than zero-training analytical Cheon et al. (2020) $f_1^{(3)}$ (`analytical_cheon`) and Remez minimax composite (`analytical_remez`) networks, because value-sorting error weights sign error by $|\Delta|$ ($e_{\text{val}} = |\Delta||1 - s(|\Delta|)|$) and stage-adaptive polynomials exploit shrinking gap distributions in later Bitonic layers.
- **Discriminating Trade-Off Prediction (H2b):** When trained purely on Value MSE without an ordering penalty, `polysort_*` may trade off small-gap slope against medium-gap value accuracy, so on **adversarial near-ties** ($\delta = 10^{-3}$), analytical Remez minimax may tie or outperform MSE-only `polysort_shared` in pairwise ordering unless a monotonicity/ordering loss term is added.
- **Falsification Criterion:** H2 is **falsified** if `polysort_adaptive` fails to reduce Uniform $[-1, 1]^n$ Value MAE by at least **15% relative** vs. `analytical_cheon` AND beat `analytical_remez` with corrected $p < 0.01$ at iso-depth ($L=3, d=3$).

### H3 — Empirical Depth Separation at Iso-Degree (Bernstein vs. Cheon)
- **Claim:** At identical comparator degree $D = 9$ inside the exact same Bitonic sorting network ($n = 8, 16$), a **composite depth-2 cubic** comparator ($L=2, d=3 \Rightarrow D=9$) achieves lower Value MAE and lower seed variance than a **flat depth-1 degree-9** polynomial ($L=1, d=9 \Rightarrow D=9$), and increasing composition depth $L \in \{1, 2, 3, 4\}$ with cubic blocks reduces Value MAE monotonically.
- **Falsification Criterion:** H3 is **falsified** if flat $(L=1, d=9)$ achieves equal or lower Value MAE than composite $(L=2, d=3)$ across 3+ seeds, OR if $(L=3, d=3)$ fails to improve over $(L=2, d=3)$ by $\ge 20\%$ relative MAE.

### H4 — Constant-Depth All-to-All Ranking (`PolyPairwiseRank`) vs. Swap Networks (`PolySortNet`)
- **Claim:** On $n = 8$, the Mazzone-inspired constant-depth `poly_pairwise_rank` (comparison depth 2, with exact polynomial tie-correction $E = 4C(1-C)$) will significantly outperform unstructured PNNs (`pinet_ncp_skip`, `poly_mlp`, `deepsets_poly`), **but** as $n$ increases ($n=16, 32$) at fixed polynomial budget, `polysort_shared` will outperform `poly_pairwise_rank` on Value MAE and exact monotonicity because `poly_pairwise_rank`'s rank indicator must resolve spacing $1/n$ without exact sum-conservation guarantees.
- **Falsification Criterion:** H4 is **falsified** if `poly_pairwise_rank` achieves equal or lower Value MAE than `polysort_shared` at $n = 16$ under matched comparator degree.

### H5 — Zero-Shot Length & Shift Generalization vs. Theoretical Scale-Boundary Blow-Up
- **Claim (Positive Invariants):** `polysort_shared` trained exclusively on $n = 8$ over $[-1, 1]^8$ will:
  1. Exhibit **exact shift equivariance** ($\|f_\theta(x + c\mathbf{1}) - (f_\theta(x) + c\mathbf{1})\|_\infty < 10^{-5}$ in float32 and $< 10^{-12}$ in float64) for $c \in \{1, 10, 100\}$.
  2. Exhibit **exact sum conservation** ($|\sum \hat{y}_i - \sum x_i| < 10^{-5}$ in float32, $< 10^{-12}$ in float64).
  3. **Generalize zero-shot** to unseen list lengths $n \in \{4, 16, 32, 64, 128\}$ with Value MAE remaining $< 0.03$ and Kendall's $\tau > 0.98$ even at $n = 128$ (16$\times$ training length).
- **Claim (Predicted Negative Boundary):** On scaled inputs $s \cdot x$ with $s > 1$, strictly pure `polysort_shared` MUST exhibit a sharp **polynomial blow-up cliff** (MAE $> 1.0$ or numerical overflow by $s = 2.0$–$5.0$), whereas the non-polynomial scale-normalized ablation (`ablation_scale_norm`) eliminates this cliff completely (MAE scales linearly with $s$, constant relative MAE).
- **Falsification Criterion:** H5 is **falsified** if shift or sum error exceeds numerical tolerance ($10^{-4}$ float32), if zero-shot $n=64$ Kendall's $\tau < 0.95$, OR if strictly pure `polysort_shared` does not diverge (MAE $< 0.1$) at $s = 5.0$.

### H6 — Parameterization & Structural Topology Causality (Ablations)
- **Claim:**
  1. Fixed-point odd Chebyshev parameterization ($h_k(u) = u + u(1-u^2)\sum c_{k,j} T_{2j}(u)$) achieves better training stability (0% NaN divergence rate) and lower Value MAE than raw monomial basis and unconstrained general (even+odd) polynomials.
  2. Replacing Bitonic/Odd-Even wiring with a **Random Fixed Wiring** negative control causes catastrophic failure (MAE $> 10\times$ worse, Kendall's $\tau < 0.70$), proving causally that the sorting-network topology carries essential algorithmic structure, while **Learned Wiring** (`polysort_learned_wire`) improves substantially over Random Wiring yet lags fixed Bitonic wiring at $n \ge 8$.
- **Falsification Criterion:** H6 is **falsified** if random fixed wiring achieves Kendall's $\tau > 0.85$ at $n=8$, or if raw monomial basis matches or beats fixed-point Chebyshev across all depths $L \in \{2, 3, 4\}$ with 0% divergence.

---

## 3. Pre-Specified Metrics

### Primary Metrics (Used for Hypothesis Testing)
1. **Value MAE ($\downarrow$):** $\mathrm{MAE}(\hat{y}, y) = \frac{1}{n}\sum_{k=1}^n |\hat{y}_k - x_{(k)}|$.
2. **Kendall's $\tau$ ($\uparrow$):** Rank correlation between $\hat{y}$ and the indices $(1, \dots, n)$ — wait: let's define ordering metrics with **100% mathematical precision** so there is ZERO ambiguity!
   - **Monotonicity Rate ($\uparrow$):** Fraction of test lists strictly/weakly sorted: $\frac{1}{N}\sum_{m=1}^N \mathbf{1}\big[\hat{y}^{(m)}_1 \le \hat{y}^{(m)}_2 \le \dots \le \hat{y}^{(m)}_n\big]$ (using strict tolerance $\hat{y}_k \le \hat{y}_{k+1} + 10^{-7}$ for float32 ties, and exact $\le$ reported alongside).
   - **Pairwise Order Accuracy ($\uparrow$):** Fraction of all $\binom{n}{2}$ pairs $(i < j)$ satisfying $\hat{y}_i \le \hat{y}_j$ (equivalently $\frac{1 + \tau_{\text{mono}}}{2}$, where $\tau_{\text{mono}} = 1 - \frac{2 \cdot \mathrm{inversions}(\hat{y})}{\binom{n}{2}}$ is the Kendall's $\tau$ of $\hat{y}$ against ascending order $(1, \dots, n)$!).
   - **Value-Matched Exact Sort Rate ($\uparrow$):** Fraction of test lists that are BOTH monotonic ($\hat{y}_1 \le \dots \le \hat{y}_n$) AND have max elementwise value error $\|\hat{y} - y\|_\infty < \varepsilon_{\text{tol}}$ (reported at $\varepsilon_{\text{tol}} \in \{0.05, 0.01, 0.001\}$). Why this dual metric? Because a trivial constant output $[0, 0, \dots, 0]$ is vacuously monotonic! Combining Monotonicity Rate with Value MAE and $\varepsilon$-Exact Sort Rate prevents metric gaming!

### Secondary & Diagnostic Metrics
- **RMSE ($\downarrow$):** $\sqrt{\frac{1}{n}\sum_{k=1}^n (\hat{y}_k - y_k)^2}$.
- **Max Elementwise Error $L_\infty$ ($\downarrow$):** $\max_{1 \le k \le n} |\hat{y}_k - y_k|$ (mean and 99th percentile across $N_{\text{test}}$).
- **Multiset Wasserstein-1 Distance $W_1(\hat{y}, y)$ ($\downarrow$):** $\frac{1}{n}\sum_{k=1}^n |\mathrm{sort}(\hat{y})_k - y_k|$ — cleanly disentangles **value distortion** (how faithful the multiset of output values is to the input values) from **ordering error**!
- **Sum Conservation Error ($\downarrow$):** $\big|\sum_{k=1}^n \hat{y}_k - \sum_{i=1}^n x_i\big|$.
- **Structural Complexity:** Learnable parameter count, effective polynomial degree $D_{\text{eff}}$, sequential multiplicative depth $d_{\text{mult}}$, forward latency (ms per 1,000 lists on GPU & CPU).
- **Empirical Scale Stability Radius $R_{\text{stable}}$ ($\uparrow$):** Largest scale factor $s \in \{0.1, 0.5, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 5.0\}$ with relative MAE ($\mathrm{MAE}/s$) $< 0.05$.

---

## 4. Data Generation, Splits & Zero-Leakage Protocol

- **Deterministic Seed-Keyed Streams:** Every split uses an independent `numpy.random.Generator` seeded via `SHA256(f"{split_name}:{n}:{seed}")`, ensuring mathematically independent datasets across `train`, `val`, `test`, and every OOD suite.
- **Sizes:**
  - `train`: $200{,}000$ i.i.d. lists from $\mathrm{Uniform}[-1, 1]^n$ (or online infinite stream? **Decision: Fixed pre-generated $200{,}000$-list training set** so ALL models — tiny 6-parameter `PolySortNet` and 100k-parameter `PiNet`/`ReLUMLP` — are trained on the **EXACT SAME finite sample budget**, enabling honest generalization & overfitting measurement, plus verifiable zero overlap with `val`/`test`!).
  - `val`: $10{,}000$ i.i.d. lists (used strictly for checkpoint selection & hyperparameter sweep).
  - `test`: $10{,}000$ i.i.d. lists (touched ONLY at final evaluation).
- **Held-Out OOD & Stress Test Suites ($N = 2{,}000$ lists each):**
  1. `ood_gaussian`: $\mathcal{N}(0, 0.35^2)$ clipped to $[-1, 1]$.
  2. `ood_bimodal`: 50/50 mixture of $\mathcal{N}(-0.7, 0.1^2)$ and $\mathcal{N}(+0.7, 0.1^2)$ clipped to $[-1, 1]$.
  3. `ood_skewed`: $2 \cdot \mathrm{Beta}(0.5, 5.0) - 1$ (heavily clustered near $-1$).
  4. `ood_integers`: Uniform discrete grid $\{-10, -9, \dots, 10\} / 10$ (frequent exact ties).
  5. `ood_duplicates_50`: 50% of entries duplicated from remaining elements.
  6. `ood_nearties_1e-2`, `ood_nearties_1e-3`, `ood_nearties_1e-4`: Lists containing planted adjacent gaps of exact size $\delta \in \{10^{-2}, 10^{-3}, 10^{-4}\}$.
  7. `ood_edge_sorted` & `ood_edge_reversed`: Already-sorted and strictly reverse-sorted lists.
  8. `ood_shift_1`, `ood_shift_10`, `ood_shift_100`: $x + c\mathbf{1}$ for $c \in \{1, 10, 100\}$.
  9. `ood_scale_s`: $s \cdot x$ for $s \in \{0.1, 0.5, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 5.0\}$.
  10. `ood_length_n`: Zero-shot list lengths $n \in \{4, 8, 16, 32, 64, 128\}$.

---

## 5. Statistical Analysis Plan

- **Multi-Seed Protocol:** All 9 core comparison models evaluated over **5 independent random seeds** (`42, 123, 456, 789, 1024`) on $n = 8$ and $n = 16$. Summary tables report `mean ± std` across seeds AND pooled test-set **95% percentile bootstrap confidence intervals** ($B = 10{,}000$ resamples).
- **Paired Hypothesis Testing:** For every preregistered comparison (H1–H6), compute per-list paired metric differences $d_m = M_A(x^{(m)}) - M_B(x^{(m)})$ on the exact same $N_{\text{test}} = 10{,}000$ test lists; run two-sided **Wilcoxon signed-rank tests** and report exact $p$-values and rank-biserial effect sizes, applying **Holm-Bonferroni** familywise error rate control at $\alpha = 0.05$ across all planned hypothesis comparisons.
- **Float64 Verification:** All primary models are evaluated in BOTH `float32` and `float64` on `test` and `ood_shift_*` to separate genuine mathematical error from `float32` roundoff noise.
- **Leakage Sanity Control:** `shuffled_target_ctrl` trains `polysort_adaptive` on randomly permuted targets; its test MAE MUST match a naive mean predictor (confirming zero target leakage).

---

## 6. Confounder Checklist

| Potential Confounder | Mitigation in This Study |
| :--- | :--- |
| **Parameter count mismatch** | Parameter-matched capacity among unstructured baselines (`pinet_ncp_skip`, `poly_mlp`, `relu_mlp`, `deepsets_poly`, `set_transformer` ~25k–65k params), while explicitly reporting parameter counts alongside accuracy so tiny structured PNNs (`polysort_shared`: 3–9 params!) are never conflated with overparameterized models. |
| **Degree / depth mismatch vs. analytical CKKS** | `polysort_shared` and `polysort_adaptive` compared against `analytical_cheon` and `analytical_remez` at **strictly identical** sorting wiring, layer count $\ell$, composition depth $L=3$, and block degree $d=3$ (total comparator degree $27$). |
| **Optimizer / LR bias** | Standard AdamW + cosine schedule + gradient clipping across all models, PLUS explicit hyperparameter sensitivity sweep in Stage 4E verifying conclusions do not depend on cherry-picked LR/weight decay. |
| **Vacuous monotonicity (constant output)** | Always co-report Value MAE, Wasserstein-1 $W_1$, Monotonicity Rate, and $\varepsilon$-Exact Sort Rate together. |
| **Hidden non-polynomial ops** | Automated PyTorch FX tracer test (`tests/test_purity.py`) rejects any non-polynomial op in any model tagged `is_pure_polynomial=True`. |

---

## 7. Pre-Hoc vs. Post-Hoc Analysis Log

- **Pre-Hoc Analyses (Locked 2026-11-25):** Hypotheses H1–H6, all 17 models/controls, all 6 ablation families, and all 10 OOD suites documented in `reports/planned-runs.md`.
- **Post-Hoc Exploratory Analyses:** *(None yet — any exploratory follow-up triggered by unexpected empirical findings will be logged here with date and rationale.)*
