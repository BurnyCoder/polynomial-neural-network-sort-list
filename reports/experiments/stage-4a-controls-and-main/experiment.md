# Stage 4A Experiment Report: Zero-Training Controls, Analytical CKKS Sorters & Primary `PolySortNet` Models

**Date:** 2026-11-25  
**Seed:** `42` (single-seed primary exploration; 5-seed statistical rigor in Stage 4E)  
**Dataset:** $200{,}000$ train / $10{,}000$ val / $10{,}000$ held-out test (`Uniform[-1, 1]^n`) + 10 OOD suites ($2{,}000$ each) + zero-shot length sweep ($n \in \{4, 8, 16, 32, 64, 128\}$).  
**Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU (`torch 2.11.0+cu128`).

---

## 1. Held-Out In-Distribution Results ($N_{\text{test}} = 10{,}000$)

### List Length $n = 8$ (Bitonic depth $\ell = 6$)

| Run ID | Pure Poly? | Params | Comp. Deg | MAE ($\downarrow$) | RMSE ($\downarrow$) | Mono Rate ($\uparrow$) | Pair Acc ($\uparrow$) | Kendall $\tau$ ($\uparrow$) | $R_{\text{stable}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ctrl_identity_n8` | Yes | 0 | 1 | 0.583411 | 0.764280 | 0.00% | 50.00% | -0.0001 | 0.0 |
| `ctrl_rand_pinet_n8` | Yes | 201,608 | 9 | 0.500190 | 0.577334 | 0.02% | 49.82% | -0.0036 | 0.0 |
| `ctrl_rand_polysort_n8` | Yes | 3 | 27 | 0.240399 | 0.288962 | 94.81% | 99.80% | 0.9961 | 0.0 |
| `ctrl_linear_n8` (OLS) | Yes | 72 | 1 | 0.134646 | 0.169388 | 99.53%* | 99.98%* | 0.9996* | 0.0 |
| `ana_cheon_f1_L3_n8` | Yes | 0 | 27 | 0.113616 | 0.139496 | 96.10% | 99.80% | 0.9961 | 0.0 |
| `ana_cheon_f2_L2_n8` | Yes | 0 | 25 | 0.107184 | 0.131642 | 96.06% | 99.80% | 0.9960 | 0.0 |
| `ana_remez_L3_n8` (Lee '22) | Yes | 0 | 27 | 0.053090 | 0.074156 | 71.60% | 98.41% | 0.9682 | 0.0 |
| `ana_remez_L3_m2_n8` | Yes | 0 | 125 | `inf` | `inf` | 85.92% | 99.18% | 0.9836 | 0.0 |
| `ana_cheon_fg_n8` ($f_3\circ g_3$) | Yes | 0 | 49 | 0.027881 | 0.035066 | 94.19% | 99.73% | 0.9946 | 1.1 |
| **`main_polysort_shared_n8`** | **Yes** | **3** | **27** | **0.037757** | **0.046529** | **90.14%** | **99.50%** | **0.9900** | **1.3** |
| **`main_polysort_adaptive_n8`** | **Yes** | **18** | **27** | **0.033254** | **0.041907** | **94.09%** | **99.75%** | **0.9949** | **1.3** |
| **`main_polysort_shared_m2_n8`** | **Yes** | **6** | **125** | **0.007941** | **0.010433** | **92.22%** | **99.67%** | **0.9935** | **1.0** |
| **`main_polysort_adaptive_m2_n8`** | **Yes** | **36** | **125** | **0.007241** | **0.009620** | **95.92%** | **99.83%** | **0.9967** | **1.0** |

*\*Note on `ctrl_linear`: Optimal OLS linear regression predicts $\hat{y}_k \approx \bar{x} + \mathbb{E}[x_{(k)}]$, a mean-shifted constant sorted template; it is almost always monotone yet suffers severe Value MAE (`0.1346`) because it cannot sort elements.*

### List Length $n = 16$ (Bitonic depth $\ell = 10$)

| Run ID | Pure Poly? | Params | Comp. Deg | MAE ($\downarrow$) | RMSE ($\downarrow$) | Mono Rate ($\uparrow$) | Pair Acc ($\uparrow$) | Kendall $\tau$ ($\uparrow$) | $R_{\text{stable}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ctrl_identity_n16` | Yes | 0 | 1 | 0.624327 | 0.790615 | 0.00% | 49.97% | -0.0006 | 0.0 |
| `ctrl_linear_n16` (OLS) | Yes | 272 | 1 | 0.098597 | 0.124860 | 99.67%* | 100.00%* | 0.9999* | 0.0 |
| `ana_cheon_f1_L3_n16` | Yes | 0 | 27 | 0.145679 | 0.179540 | 85.95% | 99.86% | 0.9972 | 0.0 |
| `ana_cheon_f2_L2_n16` | Yes | 0 | 25 | 0.136867 | 0.168912 | 86.14% | 99.86% | 0.9973 | 0.0 |
| `ana_remez_L3_n16` | Yes | 0 | 27 | 0.073997 | 0.114251 | 28.76% | 97.74% | 0.9549 | 0.0 |
| `ana_remez_L3_m2_n16` | Yes | 0 | 125 | `inf` | `inf` | 49.74% | 97.24% | 0.9447 | 0.0 |
| `ana_cheon_fg_n16` ($f_3\circ g_3$) | Yes | 0 | 49 | 0.037486 | 0.046753 | 86.22% | 99.83% | 0.9967 | 1.1 |
| **`main_polysort_shared_n16`** | **Yes** | **3** | **27** | **0.048420** | **0.059968** | **73.58%** | **99.62%** | **0.9924** | **1.3** |
| **`main_polysort_adaptive_n16`** | **Yes** | **30** | **27** | **0.036939** | **0.046879** | **75.81%** | **99.71%** | **0.9942** | **1.3** |
| **`main_polysort_shared_m2_n16`** | **Yes** | **6** | **125** | **0.011082** | **0.014053** | **72.87%** | **99.66%** | **0.9933** | **1.0** |
| **`main_polysort_adaptive_m2_n16`** | **Yes** | **60** | **125** | **0.009725** | **0.012558** | **82.45%** | **99.81%** | **0.9963** | **1.0** |

---

## 2. Zero-Shot List-Length Generalization ($n_{\text{train}} = 8 \to n_{\text{test}} \in \{4..128\}$)

| Model (Trained at $n=8$ Only) | $n=4$ MAE ($\tau$) | $n=8$ MAE ($\tau$) | $n=16$ MAE ($\tau$) | $n=32$ MAE ($\tau$) | $n=64$ MAE ($\tau$) | $n=128$ MAE ($\tau$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ana_cheon_f1_L3` (0 params) | 0.0784 (0.991) | 0.1135 (0.996) | 0.1451 (0.997) | 0.1758 (0.997) | 0.2135 (0.997) | 0.2600 (0.997) |
| `ana_cheon_fg` (0 params, deg 49) | 0.0168 (0.997) | 0.0280 (0.994) | 0.0378 (0.997) | 0.0445 (0.998) | 0.0487 (0.999) | 0.0514 (0.999) |
| `main_polysort_shared` (3 params, deg 27) | 0.0252 (0.993) | 0.0376 (0.988) | 0.0485 (0.993) | 0.0551 (0.995) | 0.0581 (0.997) | 0.0581 (0.998) |
| **`main_polysort_shared_m2` (6 params, deg 125)** | **0.0046 (0.996)** | **0.0080 (0.994)** | **0.0118 (0.993)** | **0.0156 (0.993)** | **0.0186 (0.996)** | **0.0200 (0.998)** |

---

## 3. Exact Invariants & Negative Boundary Verification (Float64 + OOD Scale Cliff)

- **Exact Float64 Sum Conservation:** `2.1e-16` – `2.2e-16` across all `PolySortNet` variants (exact machine epsilon!).
- **Exact Float64 Shift Equivariance ($x + 100\mathbf{1}$):** $L_\infty$ error `6.6e-14` – `8.4e-14`.
- **OOD Duplicates & Near-Ties ($\delta = 10^{-3}$):** `main_polysort_adaptive_m2_n8` achieves `MAE = 0.0066` on both 50% duplicates and $10^{-3}$ near-ties (actually *lower* than uniform ID!).
- **Predicted Negative Result (Polynomial Scale Blow-Up Cliff — H5):** Strictly pure `PolySortNet` diverges cleanly outside its domain:
  - Cubic ($m=1$, deg 27): stable through $s = 1.3$ ($R_{\text{stable}} = 1.3$), degrades to `MAE = 0.169` at $s = 1.5$, diverges to `inf` at $s = 2.0, 5.0$.
  - Quintic ($m=2$, deg 125): stable at $s = 1.0$ ($R_{\text{stable}} = 1.0$), diverges to `inf` at $s = 1.5, 2.0, 5.0$.
