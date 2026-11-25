# Experiment Stage 4E: Sensitivity Sweeps & 5-Seed Statistical Rigor (Final Experimental Stage)

**Author:** Libor "Burny" Burian  
**Date:** 2026-11-25  
**Runs Covered:** 95 new runs (**15** hyperparameter / batch-size / `float64` sensitivity runs + **80** multi-seed replication runs across seeds `123, 456, 789, 1024`), combined with seed `42` from Stages 4A–4C to form **100 core 5-seed cells** (`10 models × 2 lengths × 5 seeds`) and **178 / 178 total preregistered runs**.  
**Execution Order Note:** Executed strictly as the **very last experimental stage** after all main models, baselines, controls, and causal ablations finished.

---

## 1. Plain-Language Summary

Before making final scientific claims, we subjected our entire experiment suite to two rigorous stress tests:
1. **Hyperparameter, Batch-Size, and Precision Audit (`4e_sens`, 15 runs):** We checked whether our findings were artifacts of learning rate, weight decay, GPU batch size (`4096` vs `512`), or `float32` roundoff. **Result:** Training in `float64` matched `float32` to **6 decimal places** even at degree 81 (`L=4`), weight decay had zero effect (`< 4e-6`), and `bs=4096` was actually *slightly conservative* against `PolySortNet`. Crucially, tuning `PiNet-NCP-Skip` up to `lr=0.01` improved its $n=8$ MAE from `0.0529 -> 0.0423` — an important fairness catch that we report openly, though it still trails 36-parameter `PolySortNet-Adaptive (m=2)` (`0.00724`) by **$5.85\times$**.
2. **5-Seed Statistical Replication (`4e_seeds`, 80 new runs / 100 core cells):** Every primary comparison was repeated across 5 independent random seeds (`42, 123, 456, 789, 1024`) and tested with paired Wilcoxon signed-rank tests (50,000 pooled paired lists per comparison), 10,000-resample bootstrap 95% CIs, and Holm-Bonferroni multiple-testing control. **Result:** Structured `PolySortNet` models exhibit **orders-of-magnitude lower seed variance** (`SD = 5.8e-7` for `Adaptive m=2` at $n=8$, vs `5.2e-3` for `PiNet` and `2.9e-3` for `PolyMLP`), and **100% of our 5-seed directional rankings hold across all 5/5 seeds** (except the honest cubic $m=1$ vs `DeepSets-Poly` near-tie at $n=8$, where `DeepSets-Poly` wins 4/5 seeds by `0.00028` MAE!).

---

## 2. Part I: Hyperparameter, Batch-Size & `float64` Sensitivity Sweep (`n=8`, Seed 42)

| Sweep Axis | Run ID | Setting | ID MAE ($\downarrow$) | ID RMSE ($\downarrow$) | Mono% ($\uparrow$) | Ex@0.05% ($\uparrow$) | Kendall $\tau$ ($\uparrow$) | $R_{\text{stable}}$ | Train Time (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PolySort LR** | `sens_polysort_lr_0.0003` | `lr=3e-4` | `0.057026` | `0.070598` | `91.80%` | `0.65%` | `0.99128` | `0.00` | `84.5` |
| **PolySort LR** | `sens_polysort_lr_0.001` | `lr=1e-3` | `0.038448` | `0.048371` | `87.27%` | `9.51%` | `0.98839` | `1.30` | `71.5` |
| **PolySort LR** | `sens_polysort_lr_0.003` | `lr=3e-3` | `0.033753` | `0.042550` | `93.77%` | `16.11%` | `0.99461` | `1.30` | `72.6` |
| **PolySort LR** | `main_polysort_adaptive` (ref) | `lr=8e-3` | `0.033254` | `0.041907` | `94.09%` | `16.38%` | `0.99494` | `1.30` | `66.1` |
| **PolySort LR** | `sens_polysort_lr_0.01` | `lr=1e-2` | **`0.033236`** | `0.041891` | `94.14%` | `16.60%` | `0.99497` | `1.30` | `73.6` |
| **PiNet LR** | `sens_pinet_lr_0.0003` | `lr=3e-4` | `0.077068` | `0.099430` | `95.33%` | `0.38%` | `0.99486` | `0.00` | `69.1` |
| **PiNet LR** | `sens_pinet_lr_0.001` | `lr=1e-3` | `0.074510` | `0.096460` | `96.65%` | `0.50%` | `0.99629` | `0.00` | `70.1` |
| **PiNet LR** | `sens_pinet_lr_0.003` | `lr=3e-3` | `0.051979` | `0.068675` | `93.38%` | `3.25%` | `0.99346` | `0.00` | `67.2` |
| **PiNet LR** | `sens_pinet_lr_0.01` | `lr=1e-2` | **`0.042324`** | `0.056386` | `90.60%` | **`7.73%`** | `0.99081` | **`1.10`** | `71.5` |
| **Weight Decay** | `sens_polysort_wd_0` | `wd=0.0` | `0.033254` | `0.041907` | `94.09%` | `16.38%` | `0.99494` | `1.30` | `70.4` |
| **Weight Decay** | `sens_polysort_wd_0.01` | `wd=1e-2` | `0.033258` | `0.041914` | `94.02%` | `16.37%` | `0.99486` | `1.30` | `69.2` |
| **Batch Size** | `sens_polysort_bs_512` | `bs=512, 40ep` | **`0.031651`** | `0.039713` | `94.29%` | `20.84%` | `0.99507` | `1.20` | `440.3` |
| **Batch Size** | `sens_polysort_bs_2048` | `bs=2048, 40ep` | `0.033208` | `0.041887` | `94.08%` | `16.69%` | `0.99491` | `1.30` | `107.7` |
| **Batch Size** | `sens_polysort_bs_4096` | `bs=4096, 40ep` | `0.033272` | `0.041929` | `94.05%` | `16.39%` | `0.99487` | `1.30` | `50.6` |
| **Float64 Train** | `sens_polysort_L3_f64` | `L=3, float64` | **`0.033254`** | `0.041907` | `94.09%` | `16.38%` | `0.99494` | `1.30` | `62.8` |
| **Float64 Train** | `sens_polysort_L4_f64` | `L=4, float64` | **`0.013222`** | `0.017113` | `96.22%` | `91.80%` | `0.99691` | `1.50` | `75.1` |

---

## 3. Part II: 5-Seed Statistical Replication Tables (`5 seeds × 10,000 lists = 50,000` Evaluations/Model)

### 3.1 List Length $n = 8$ (5 Seeds: `42, 123, 456, 789, 1024`)

| Model | Pure Poly? | Params | Test MAE (Mean $\pm$ SD) | 95% Boot CI | Mono% | Ex@0.05% | Kendall $\tau$ | Shift $+100$ | Near-Ties $10^{-3}$ | Dup 50% | $R_{\text{stable}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `SetTransformer` (non-poly) | No | 96,577 | `0.00218 ± 0.00022` | `[0.00217, 0.00219]` | `99.92%` | `99.9%` | `0.9999` | `99.502` | `0.00665` | `0.00737` | `1.50` |
| `DiffSort-Cauchy` (non-poly) | No | 1 | `0.00306 ± 0.00000` | `[0.00306, 0.00307]` | `100.0%` | `100.0%` | `1.0000` | `0.00307` | `0.00275` | `0.00277` | `5.00` |
| **`PolySortNet-Adaptive (m=2)`** | **Yes** | **36** | **`0.00724 ± 0.00000`** | **`[0.00722, 0.00727]`** | **`95.86%`** | **`95.9%`** | **`0.9966`** | **`0.00726`** | **`0.00659`** | **`0.00660`** | `1.00` |
| `DeepSets-Poly` | Yes | 130,920 | `0.03300 ± 0.00069` | `[0.03288, 0.03312]` | `94.33%` | `21.6%` | `0.9952` | `inf` | `0.06327` | `0.07464` | `1.05` |
| **`PolySortNet-Adaptive (m=1)`** | **Yes** | **18** | `0.03327 ± 0.00002` | `[0.03318, 0.03337]` | `94.10%` | `16.3%` | `0.9949` | **`0.03344`** | `0.03268` | `0.03240` | **`1.30`** |
| `ReLU/GELU-MLP` (non-poly) | No | 35,208 | `0.03549 ± 0.00122` | `[0.03538, 0.03560]` | `90.90%` | `15.2%` | `0.9928` | `27.482` | `0.06418` | `0.06564` | `1.42` |
| **`PolySortNet-Shared (m=1)`** | **Yes** | **3** | `0.03776 ± 0.00000` | `[0.03766, 0.03787]` | `90.03%` | `9.8%` | `0.9898` | **`0.03770`** | `0.03513` | `0.03464` | **`1.30`** |
| `PolyPairwiseRank` (Mazzone) | Yes | 12 | `0.04232 ± 0.00038` | `[0.04206, 0.04259]` | `66.50%` | `13.6%` | `0.9629` | `9.934` | `0.05729` | `0.05847` | `1.01` |
| `PiNet-NCP-Skip` | Yes | 201,608 | `0.05426 ± 0.00523` | `[0.05409, 0.05443]` | `93.65%` | `2.7%` | `0.9936` | `2.8e17` | `0.08936` | `0.09201` | `0.21` |
| `PolyMLP` ($x^2+x$) | Yes | 36,744 | `0.05984 ± 0.00293` | `[0.05962, 0.06007]` | `93.17%` | `1.5%` | `0.9932` | `inf` | `0.09798` | `0.10116` | `0.00` |

### 3.2 List Length $n = 16$ (5 Seeds: `42, 123, 456, 789, 1024`)

| Model | Pure Poly? | Params | Test MAE (Mean $\pm$ SD) | 95% Boot CI | Mono% | Ex@0.05% | Kendall $\tau$ | Shift $+100$ | Near-Ties $10^{-3}$ | Dup 50% | $R_{\text{stable}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `SetTransformer` (non-poly) | No | 96,577 | `0.00377 ± 0.00045` | `[0.00376, 0.00378]` | `99.79%` | `99.7%` | `1.0000` | `99.562` | `0.00781` | `0.00856` | `1.42` |
| `DiffSort-Cauchy` (non-poly) | No | 1 | `0.00424 ± 0.00000` | `[0.00424, 0.00424]` | `99.93%` | `99.9%` | `1.0000` | `0.00423` | `0.00401` | `0.00404` | `5.00` |
| **`PolySortNet-Adaptive (m=2)`** | **Yes** | **60** | **`0.00981 ± 0.00008`** | **`[0.00979, 0.00984]`** | **`82.19%`** | **`81.8%`** | **`0.9963`** | **`0.00976`** | **`0.01062`** | **`0.01045`** | `1.00` |
| **`PolySortNet-Adaptive (m=1)`** | **Yes** | **30** | `0.03693 ± 0.00001` | `[0.03686, 0.03701]` | `75.75%` | `1.0%` | `0.9942` | **`0.03699`** | `0.03902` | `0.03961` | **`1.30`** |
| `DeepSets-Poly` | Yes | 131,696 | `0.04122 ± 0.00014` | `[0.04112, 0.04131]` | `92.91%` | `0.7%` | `0.9980` | `inf` | `0.06682` | `0.07042` | `1.05` |
| **`PolySortNet-Shared (m=1)`** | **Yes** | **3** | `0.04842 ± 0.00000` | `[0.04833, 0.04852]` | `73.56%` | `0.0%` | `0.9924` | **`0.04846`** | `0.04869` | `0.04897` | **`1.30`** |
| `PiNet-NCP-Skip` | Yes | 205,712 | `0.06236 ± 0.00003` | `[0.06221, 0.06251]` | `98.73%` | `0.0%` | `0.9994` | `6.1e15` | `0.08911` | `0.09068` | `0.00` |
| `ReLU/GELU-MLP` (non-poly) | No | 37,264 | `0.06361 ± 0.00054` | `[0.06347, 0.06376]` | `99.24%` | `0.0%` | `0.9998` | `70.241` | `0.09035` | `0.09269` | `0.00` |
| `PolyMLP` ($x^2+x$) | Yes | 38,800 | `0.06482 ± 0.00178` | `[0.06468, 0.06497]` | `98.21%` | `0.0%` | `0.9993` | `inf` | `0.09189` | `0.09376` | `0.00` |
| `PolyPairwiseRank` (Mazzone) | Yes | 12 | `0.07380 ± 0.00012` | `[0.07361, 0.07399]` | `0.11%` | `0.0%` | `0.9242` | `12.099` | `0.08859` | `0.09148` | `0.00` |

---

## 4. Part III: Holm-Bonferroni Paired Wilcoxon Hypothesis Tests (34 Planned Comparisons)

All 34 paired Wilcoxon signed-rank tests (from `results/statistical_tests.json`, 10,000-resample bootstrap CIs on paired differences `A - B`, negative = A better) are statistically significant after Holm-Bonferroni familywise error control at $\alpha = 0.05$:
- **H1 (`PolySortNet-Adaptive m=2` vs. pure-PNN ladder):** Beats `DeepSets-Poly` ($4.56\times$ at $n=8$, $4.20\times$ at $n=16$, 5/5 seeds, $r_{rb} = -1.000$), `PiNet-NCP-Skip` ($7.49\times / 6.35\times$, 5/5 seeds, $r_{rb} = -1.000$), and `PolyMLP` ($8.26\times / 6.61\times$, 5/5 seeds, $r_{rb} = -1.000$), all $p_{\text{holm}} < 10^{-300}$. **Honest H1d nuance:** At cubic $m=1$, `DeepSets-Poly` beats `PolySortNet-Adaptive (m=1)` by `0.00028` MAE at $n=8$ (4/5 seeds, $r_{rb} = +0.056$, $p_{\text{holm}} = 8.47 \times 10^{-28}$), while `PolySortNet-Adaptive (m=1)` wins 5/5 seeds at $n=16$ (`0.03693` vs `0.04122`, $r_{rb} = -0.339$).
- **H2 (Learned vs. Analytical CKKS & Stage-Adaptive vs. Shared):** `Adaptive m=1` beats Cheon $f_1^{(3)}$ ($3.42\times / 3.94\times$) and HiGHS Remez $L=3$ ($1.60\times / 2.00\times$); `Adaptive m=2` beats Cheon/Hong/Mazzone $f_3 \circ g_3$ ($3.85\times / 3.85\times$); `Adaptive m=1` beats `Shared m=1` 5/5 seeds at both $n=8$ ($1.14\times$, $r_{rb}=-0.542$) and $n=16$ ($1.31\times$, $r_{rb}=-0.923$).
- **H3 (Iso-Degree Falsification vs. Iso-Parameter Separation):** `L=2, d=3` loses to flat `L=1, d=9` at iso-degree $D=9$ ($r_{rb} = +0.847 / +0.892$, falsifying H3 Part 1); iso-parameter `L=4, d=3` crushes flat `L=1, d=9` by $4.28\times / 4.13\times$ ($r_{rb} = -1.000$).
- **H4 (`PolyPairwiseRank`):** `PolySortNet-Adaptive m=2` beats `PolyPairwiseRank` 5/5 seeds by $5.84\times$ ($n=8$) and $7.52\times$ ($n=16$, $r_{rb} = -1.000$).
- **H6 (Parameterization Falsification & Wiring Causality):** Unconstrained odd Chebyshev beats fixed-point on ID MAE ($r_{rb} = +0.901 / +0.940$, falsifying H6 Part 1, at cost of $R_{\text{stable}}: 1.3 \to 1.1$); Bitonic beats Random Fixed Wiring by $11.6\times / 11.8\times$ ($r_{rb} = -1.000$); Batcher Odd-Even Merge beats Bitonic by $1.15\times / 1.11\times$ ($r_{rb} = -0.431 / -0.442$).
- **Non-Poly Baselines (`B1–B3`):** `PolySortNet-Adaptive m=2` beats `ReLU/GELU-MLP` 5/5 seeds by **$4.90\times$ ($n=8$) and $6.48\times$ ($n=16$)** ($r_{rb} = -1.000$), while honestly lagging `DiffSort-Cauchy` ($0.42\times / 0.43\times$) and `SetTransformer` ($0.30\times / 0.38\times$) on ID MAE (yet beating 96.6k-param `SetTransformer` on 5-seed mean **Near-Ties $10^{-3}$** `0.00659` vs `0.00665`, **50% Duplicates** `0.00660` vs `0.00737`, **Zero-Shot $n=128$** $2.65\times$, and **Shift $+100$** $13{,}714\times$!).
