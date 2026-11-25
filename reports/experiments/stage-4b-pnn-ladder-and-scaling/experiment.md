# Stage 4B Experiment Report: Full Pure-PNN Architecture Ladder & Length Scaling ($n = 4, 8, 16, 32$)

**Date:** 2026-11-25  
**Seed:** `42` (single-seed ladder; 5-seed statistical confirmation in Stage 4E)  
**Scope:** Comparing all 9 strictly pure-polynomial neural architectures (verified 100% pure via PyTorch FX) across $n \in \{4, 8, 16, 32\}$, plus deep diagnostics (Multiset $W_1$, $\varepsilon$-Exact Sort Rate, OOD Shift $+10$, Gaussian, and 50% Duplicates).

---

## 1. Full Pure-PNN Architecture Ladder ($N_{\text{test}} = 10{,}000$)

### List Length $n = 8$

| Model (All 100% Pure Poly) | Params | MAE ($\downarrow$) | $W_1$ ($\downarrow$) | $L_\infty$ Mean | Ex@0.05 ($\uparrow$) | Ex@0.01 ($\uparrow$) | Sum Err | Shift $+10$ MAE | Dup 50% MAE | $R_{\text{stable}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`polysort_adaptive_m2`** | **36** | **0.0072** | **0.0072** | **0.0166** | **95.9%** | **16.8%** | **1.2e-7** | **0.0073** | **0.0066** | 1.0 |
| **`polysort_shared_m2`** | **6** | **0.0079** | **0.0079** | **0.0180** | **92.2%** | **10.3%** | **1.2e-7** | **0.0079** | **0.0066** | 1.0 |
| `deepsets_poly` | 130,920 | 0.0329 | 0.0327 | 0.0765 | 21.8% | 0.0% | 7.3e-3 | `inf` | 0.0827 | 1.05 |
| `polysort_adaptive` ($m=1$) | 18 | 0.0333 | 0.0332 | 0.0720 | 16.4% | 0.0% | 1.2e-7 | 0.0334 | 0.0324 | **1.3** |
| `polysort_shared` ($m=1$) | 3 | 0.0378 | 0.0377 | 0.0807 | 11.4% | 0.0% | 1.2e-7 | 0.0378 | 0.0346 | **1.3** |
| `poly_pairwise_rank` (Mazzone) | 12 | 0.0430 | 0.0408 | 0.1008 | 13.0% | 0.0% | 4.9e-2 | 1.02 | 0.0596 | 1.05 |
| `pinet_ncp_skip` (Chrysos) | 201,608 | 0.0529 | 0.0525 | 0.1224 | 3.0% | 0.0% | 9.7e-3 | `2.34e+08` | 0.0910 | 0.0 |
| `poly_mlp` (Kileel) | 36,744 | 0.0641 | 0.0637 | 0.1449 | 1.1% | 0.0% | 1.3e-2 | `8.40e+17` | 0.1046 | 0.0 |
| `polysort_learned_wire` | 466 | 0.0967 | 0.0966 | 0.2187 | 0.0% | 0.0% | 2.7e-2 | `inf` | 0.1373 | 0.0 |
| `ctrl_linear` (OLS deg 1) | 72 | 0.1346 | 0.1346 | 0.2870 | 0.0% | 0.0% | 5.4e-8 | 2.66 | 0.1977 | 0.0 |

### List Length $n = 16$

| Model (All 100% Pure Poly) | Params | MAE ($\downarrow$) | $W_1$ ($\downarrow$) | $L_\infty$ Mean | Ex@0.05 ($\uparrow$) | Sum Err | Shift $+10$ MAE | Dup 50% MAE | $R_{\text{stable}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`polysort_adaptive_m2`** | **60** | **0.0097** | **0.0097** | **0.0264** | **82.0%** | **2.2e-7** | **0.0097** | **0.0104** | 1.0 |
| **`polysort_shared_m2`** | **6** | **0.0111** | **0.0110** | **0.0290** | **72.5%** | **2.2e-7** | **0.0111** | **0.0102** | 1.0 |
| `polysort_adaptive` ($m=1$) | 30 | 0.0369 | 0.0368 | 0.0963 | 0.9% | 2.2e-7 | 0.0370 | 0.0396 | **1.3** |
| `deepsets_poly` | 131,696 | 0.0413 | 0.0412 | 0.1138 | 0.7% | 7.2e-3 | `inf` | 0.0692 | 1.05 |
| `polysort_shared` ($m=1$) | 3 | 0.0484 | 0.0483 | 0.1173 | 0.2% | 2.2e-7 | 0.0484 | 0.0472 | **1.3** |
| `pinet_ncp_skip` | 205,712 | 0.0624 | 0.0624 | 0.1679 | 0.0% | 1.2e-2 | `6.73e+06` | 0.0909 | 0.0 |
| `poly_mlp` | 38,800 | 0.0653 | 0.0653 | 0.1758 | 0.0% | 1.4e-2 | `8.68e+22` | 0.0938 | 0.0 |
| `poly_pairwise_rank` | 12 | 0.0738 | 0.0546 | 0.2294 | 0.0% | 1.8e-1 | 1.21 | 0.0915 | 0.0 |
| `polysort_learned_wire` | 2,846 | 0.0753 | 0.0753 | 0.1957 | 0.0% | 2.7e-2 | `inf` | 0.1091 | 0.0 |
| `ctrl_linear` (OLS deg 1) | 272 | 0.0986 | 0.0986 | 0.2396 | 0.0% | 8.7e-8 | 3.20 | 0.1433 | 0.0 |

---

## 2. Scaling Across List Lengths ($n = 4, 8, 16, 32$)

| Model | $n = 4$ MAE | $n = 8$ MAE | $n = 16$ MAE | $n = 32$ MAE | Adaptive vs. Shared Gain ($m=1$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `polysort_shared` ($m=1$, 3 params) | 0.0247 | 0.0378 | 0.0484 | 0.0548 | — |
| `polysort_adaptive` ($m=1$, $3\ell$ params) | 0.0231 | 0.0333 | 0.0369 | **0.0375** | **-6.2% $\to$ -11.9% $\to$ -23.7% $\to$ -31.5%** |
| `deepsets_poly` (~131k params) | **0.0121** | 0.0329 | 0.0413 | — | *(Fails Shift OOD & duplicates)* |
| `pinet_ncp_skip` (~202k params) | 0.0246 | 0.0529 | 0.0624 | — | — |
| `poly_mlp` (~37k params) | 0.0215 | 0.0641 | 0.0653 | — | — |
