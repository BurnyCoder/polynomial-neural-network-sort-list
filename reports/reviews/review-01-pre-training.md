# Adversarial Self-Review #1: Pre-Training Audit (NeurIPS-Style Red Team)

**Date:** 2026-11-25  
**Reviewer Role:** Skeptical NeurIPS Area Chair & Methodology Auditor  
**Scope:** Codebase (`src/pnn_sort/`), test suite (`tests/` — 62/62 passed), preregistration (`reports/preregistration.md`), and run matrix (`reports/planned-runs.md`) prior to any model training.

---

## 1. Summary of Pre-Training Verification
- **PyTorch FX Pure-Polynomial Audit:** 18/18 pure-PNN models (and zero-shot length views `n=4, 16, 32`) passed mechanical FX graph verification confirming zero non-polynomial ops; 5/5 non-polynomial baselines (`diffsort_*`, `relu_mlp`, `polysort_scale_norm`) were verified to fail the pure-polynomial gate as required.
- **Symbolic & Exact Invariant Audit:** Cheon $f_1\text{--}f_4$ Chebyshev expansions match SymPy exact algebra ($< 10^{-13}$); Bitonic, Odd-Even Merge, and Odd-Even Transposition wirings pass Knuth Zero-One exact verification across $n \in \{2..32\}$; `PolySortNet` satisfies float64 sum conservation ($< 10^{-13}$), shift equivariance up to $c=100$ ($< 10^{-11}$), negation anti-symmetry ($< 10^{-13}$), and constant idempotency ($< 10^{-14}$).
- **Anti-Leakage & Reproducibility:** SHA-256 keyed disjoint split hashes pass; interrupted-and-resumed training matches uninterrupted training bit-for-bit (`atol=1e-6`).

---

## 2. Adversarial Critiques Raised & Mitigations Applied

### Critique 1: "Is Cheon $f_1^{(3)}$ a strawman analytical baseline?"
- **Assessment:** **VALID — FIXED BEFORE TRAINING.** Initial unit tests showed repeated Cheon $f_1^{(3)}$ has origin slope $1.5^3 = 3.375$, yielding MAE $\approx 0.114$ on $n=8$ Uniform $[-1,1]^8$. Comparing trained networks solely against $f_1^{(3)}$ would exaggerate gains.
- **Mitigation:** Added three far stronger zero-training analytical CKKS baselines:
  1. `analytical_cheon_fg`: exact Cheon/Hong/Mazzone mixed $f_3 \circ g_3$ (degree 49, MAE $< 0.03$).
  2. `analytical_remez` ($L=3, m=1$, $\delta=0.10$, degree 27): exact HiGHS convex-LP Lee et al. (2022) minimax composite (MAE $\approx 0.044$).
  3. `analytical_remez` ($L=3, m=2$, $\delta=0.03$, degree 125): iso-degree analytical rival for `polysort_shared_m2` and `polysort_adaptive_m2`.

### Critique 2: "Could `DiffSortBaseline` be crippled by a small temperature?"
- **Assessment:** **MITIGATED.** `DiffSortBaseline` learns `log_steepness` up to $\exp(8.0) \approx 2980$ with `lr=2e-2`, allowing it to sharpen arbitrarily close to a hard comparator (non-polynomial upper bound). Note that `DiffSortBaseline` is initialized at `steepness=10.0` (standard `diffsort` default).

### Critique 3: "Are unstructured PNNs (`pinet_ncp_skip`, `deepsets_poly`, `poly_mlp`) starved for parameters?"
- **Assessment:** **MITIGATED.** Parameter counts verified:
  - `polysort_shared`: **3** params; `polysort_adaptive`: **18–30** params; `poly_pairwise_rank`: **12** params.
  - `poly_mlp`: **36.7k–38.8k** (matched to `relu_mlp`: **35.2k–37.3k**).
  - `deepsets_poly`: **130.9k–131.7k**; `pinet_ncp_skip`: **201.6k–205.7k**; `set_transformer`: **96.6k**.
  Unstructured models receive $1{,}000\times$–$60{,}000\times$ *more* parameters than `PolySortNet`, making any `PolySortNet` victory overwhelmingly attributable to architectural inductive bias rather than capacity advantage.

### Critique 4: "Can `test` split differ across models and invalidate paired Wilcoxon tests?"
- **Assessment:** **MITIGATED.** `evaluate_model_full` fixes `seed=9999` for `test` ($N=10{,}000$) and all OOD suites ($N=2{,}000$) across EVERY model and training seed, saving exact per-list metric arrays (`test_per_list.npz`) so paired Wilcoxon signed-rank tests operate on the exact same held-out lists.

### Critique 5: "Risk of vacuous monotonicity or hidden float32 roundoff."
- **Assessment:** **MITIGATED.** Every report co-logs Value MAE, RMSE, $L_\infty$, Multiset $W_1$, Sum Conservation Error, exact & tolerance Monotonicity Rate, Pairwise Order Accuracy / Kendall's $\tau$, and $\varepsilon$-Exact Sort Rates ($\varepsilon \in \{0.05, 0.01, 0.001\}$), plus a separate `float64` verification pass on `test` and `ood_shift_100`.

### Critique 6: "Is `SetTransformerBaseline` truly permutation-invariant and strong?"
- **Assessment:** **VERIFIED.** Uses 2 Pre-LN Self-Attention Blocks without positional encodings on inputs (strictly permutation-equivariant), followed by quantile-query cross-attention (strictly permutation-invariant output order statistics, length-generalizable).

---

## 3. Verdict
**APPROVED FOR STAGE 4 EXECUTION.** No methodological flaws or leakage vectors remain. Proceed to Stage 4A (zero-training controls/analytical baselines + primary `PolySortNet` models) with live dashboard telemetry and incremental GitHub/Hugging Face syncs.
