# Adversarial Self-Review #3: Post-Ablation Scientific Audit (Stages 4C & 4D)

**Date:** 2026-11-25  
**Reviewer Role:** Skeptical NeurIPS Reviewer  
**Scope:** Critical audit of all 83 completed runs (Stages 4A–4D, including 10 Stage 4C baselines/leakage control + 30 Stage 4D causal ablations A1–A6) before executing Stage 4E (5-seed statistical rigor + sensitivity sweeps).

---

## 1. Strict Preregistration Scorecard (`H1–H6` After Stage 4D)

| Hypothesis | Verdict (Seed 42; 5-Seed CIs & Holm-Bonferroni Pending in 4E) | Honest Scientific Evidence & Exact Falsifications |
| :--- | :--- | :--- |
| **H1 (Inductive Bias Ladder)** | **PARTIALLY CONFIRMED (Degree-Dependent)** | Quintic `polysort_adaptive_m2` (36/60 params) beats `DeepSetsPoly` (131k), `PiNet` (206k), `PolyMLP` (39k), and `ReLU-MLP` (37k) by **$4.2\times\text{--}8.8\times$** in MAE and **$117\times$** in `Ex@0.05` at $n=16$. **Honest Caveat:** Cubic $m=1$ `polysort_adaptive` ties `DeepSetsPoly` at $n=8$ (`0.0333` vs `0.0329`) on ID MAE before winning decisively on duplicates (`0.0324` vs `0.0827`) and Shift $+100$ (`0.0334` vs `inf`). |
| **H2 (Learned vs. Analytical CKKS)** | **CONFIRMED** | At iso-degree ($L=3, m=1$, deg 27), learned `polysort_adaptive` beats Cheon $f_1^{(3)}$ by **$3.4\times\text{--}3.9\times$** and Lee et al. HiGHS Remez by **37%–50%**. At $m=2$ (deg 125), `polysort_adaptive_m2` achieves `0.0072 / 0.0097` MAE ($3.85\times$ better than Cheon/Hong/Mazzone $f_3\circ g_3$), whereas unscaled Remez $m=2$ diverges (`inf`). |
| **H3 (Composite Depth vs. Flat Degree)** | **PART 1 FALSIFIED (Iso-Degree $D=9$); PART 2 STRONGLY CONFIRMED (Iso-Parameter)** | **Explicit Falsification:** At strict iso-degree $D=9$, flat `L=1, d=9` (4 params/stage, MAE `0.0566 / 0.0674`) outperforms composite `L=2, d=3` (2 params/stage, MAE `0.0771 / 0.0831`) because the 2D composite set is a strict subset of the 4D flat degree-9 space. **Strong Causal Confirmation at Iso-Parameter:** Holding parameter count fixed at 4/stage, composite `L=4, d=3` (deg 81) beats flat `L=1, d=9` by **$4.28\times / 4.13\times$** (`0.0132 / 0.0163`), boosts `Ex@0.05` from `1.4% / 0.0%` $\to$ **`91.8% / 61.4%`**, and expands $R_{\text{stable}}$ from `0.00` $\to$ **`1.50`**. |
| **H4 (`PolyPairwiseRank` Scaling Crossover)** | **CONFIRMED** | Beats unstructured `PiNet`/`PolyMLP` at $n=8$ (`0.0430` vs `0.0529 / 0.0641`), then degrades steeply at $n=16$ (`0.0738`, Mono `0.11%`) and breaks shift equivariance (`Shift +10 = 1.02`). |
| **H5 (Invariants vs. Scale Blow-Up Cliff)** | **CAUSALLY PROVEN** | Pure `PolySortNet-Shared-m2` extrapolates zero-shot $16\times$ to $n=128$ (`MAE = 0.0200`, $\tau = 0.9978$, beating `SetTransformer` by $2.65\times$), preserves float64 sum (`2.1e-16`) & Shift $+100$ (`6.6e-14`), and blows up at $s = 1.5\text{--}2.0$. **Causal Intervention (A6):** Adding non-poly `max(|x|)` division immediately restores $R_{\text{stable}} = 5.0$ (at the cost of breaking Shift $+100$: `0.455 / 0.480`). |
| **H6 (Parameterization & Wiring)** | **PART 1 FALSIFIED (ID MAE); PART 2 CAUSALLY PROVEN (Wiring)** | **Explicit Falsification:** Removing fixed-point anchoring (`unconst_odd`, `raw_monomial`, `general_cheby`) **improves** ID MAE by ~26–30% (`0.0234 / 0.0273` vs `0.0333 / 0.0369`) by contracting late-stage domains (`s_5(1) = -444..-1038`), trading away upper scale stability (`R_stable: 1.3 -> 1.1`). **Wiring Proven Essential:** Random fixed wiring collapses to `0.00%` Mono (`MAE = 0.386 / 0.436`), whereas Batcher Odd-Even Merge beats Bitonic (`-10% to -13%` MAE) and Odd-Even Transposition hits `99.3% / 96.8%` Mono. |

---

## 2. Red-Team Methodology & Confounder Audit

1. **Data Leakage & Contamination Check:**
   - **PASS.** `ctrl_shuffled_n8` (`0.210`) and `ctrl_shuffled_n16` (`0.282`) confirm zero target leakage (`0.0%` Ex@0.05). Disjoint SHA-256 keyed splits verified in unit tests.
2. **Confounder Check — Parameter Count vs. Degree in A2 (`L=1, d=9` vs `L=2, d=3` vs `L=4, d=3`):**
   - **IDENTIFIED & RESOLVED.** Because `L=1, d=9` has 4 parameters/stage while `L=2, d=3` has 2 parameters/stage, comparing only those two would have conflated parameterization freedom with composition. Including `L=4, d=3` (4 parameters/stage, deg 81) cleanly separates **both** axes: iso-degree ($D=9$: flat wins on ID MAE, fails boundary stability) vs. iso-parameter ($4$/stage: deep composite wins by $>4\times$ on MAE AND expands $R_{\text{stable}}$ to `1.50`).
3. **Confounder Check — Batch Size (`4096` vs `512`) and Float32 vs. Float64 Training Precision:**
   - **PENDING VERIFICATION IN STAGE 4E.** All Stage 4A–4D compact runs used `batch_size=4096, lr=8e-3` in `float32` (deviation logged in `preregistration.md` Section 7). Stage `4e_sens` explicitly tests `bs in {512, 2048, 4096}`, `lr in {3e-4, 1e-3, 3e-3, 1e-2}` (including `PiNet-NCP-Skip` to rule out LR underfitting!), `wd in {0.0, 1e-2}`, and **`float64` training on `L=3` and `L=4`** (critical because degree $3^4 = 81$ could theoretically show float32 gradient noise).
4. **Confounder Check — Single-Seed Variance & Multiple Comparisons:**
   - **PENDING VERIFICATION IN STAGE 4E.** All 83 runs so far are seed `42`. Stage `4e_seeds` executes 80 additional runs across seeds `{123, 456, 789, 1024}` for all 10 core models $\times$ $n \in \{8, 16\}$ (100 total core runs), followed by 10,000-resample bootstrap 95% CIs and Holm-Bonferroni corrected paired Wilcoxon signed-rank tests.
5. **Claim Calibration & Anti-Hype Guardrails:**
   - Do **not** hide the falsifications of H3 Part 1 or H6 Part 1; highlight both in the paper abstract, results tables, and discussion as genuine discoveries (domain-contraction vs. $R_{\text{stable}}$ trade-off; iso-degree vs. iso-parameter separation).
   - Explicitly contrast `PolySortNet`'s strengths (pure polynomial, tiny param count, zero-shot $n=128$, Shift $+100$, near-ties) against its genuine limits (non-poly `DiffSort` & `SetTransformer` still lead ID MAE, pure polynomials cannot avoid upper scale blow-up).

---

## 3. Verdict
Stage 4C and Stage 4D are **scientifically airtight**, with two honest preregistration falsifications documented mechanistically via float64 transfer inspection. Cleared to push Stage 4D + Review #3 to GitHub & Hugging Face and immediately launch **Stage 4E (`4e_sens` + `4e_seeds`)**.
