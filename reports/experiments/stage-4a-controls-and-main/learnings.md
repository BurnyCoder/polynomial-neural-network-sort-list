# Stage 4A Learnings: Mechanistic Analysis of Learned vs. Analytical Polynomials

**Date:** 2026-11-25

## 1. What Did Gradient Descent Actually Discover?
Inspecting the exact learned weights of `main_polysort_shared_n8_s42` (`[[1.4017], [0.8521], [0.7530]]`) and `main_polysort_adaptive_n8_s42` reveals two deep mechanistic phenomena:

1. **Automatic Rediscovery & Generalization of Cheon's Mixed $f \circ g$ Principle:**
   - Analytical Cheon $f_1^{(3)}$ fixes $c_1 = c_2 = c_3 = 0.5$, yielding small-gap slope $1.5^3 = 3.375$.
   - End-to-end training spontaneously learns a **decreasing-gain schedule** ($c_1 = 1.40 > c_2 = 0.85 > c_3 = 0.75$): Layer 1 amplifies small gaps aggressively ($h_1'(0) = 2.40$, allowing controlled intermediate overshoot like Cheon's $g_3$), while Layers 2–3 contract values back toward $\pm 1$, lifting total small-gap slope to **$7.80\times$** ($2.31\times$ steeper than Cheon $f_1^{(3)}$ at identical degree 27!) and cutting Value MAE by **$3.0\times$**.
2. **Stage-Adaptive Sharpening Across Sorting Network Depth:**
   - In `main_polysort_adaptive_n8_s42`, total origin slope increases from **$6.78$** at Bitonic Stage 0 (comparing wide random gaps) to **$9.57$** at Bitonic Stage 5 (final merge of adjacent nearly-sorted elements).
   - This stage specialization cuts Value MAE by another **12.0%** at $n=8$ (`0.0378 -> 0.0333`) and **23.7%** at $n=16$ (`0.0484 -> 0.0369`), while simultaneously raising full-list monotonicity from `90.14% -> 94.09%` ($n=8$) and `72.87% -> 82.45%` ($n=16$, $m=2$).
3. **Why Unscaled Analytical Remez $m=2$ Diverged (`MAE = inf`) While Trained `m=2` Succeeded (`MAE = 0.0072`):**
   - Analytical Lee et al. Remez minimax optimizes a *single* comparator gate in isolation; when $\tau_L > 0.05$ compounds across 6–10 Bitonic layers, small boundary overshoots cascade outside $[-1, 1]$ and explode!
   - End-to-end backpropagation through the full Bitonic sorting network **automatically enforces multi-stage stability**, yielding `0.00724` MAE on $n=8$, `0.00973` MAE on $n=16$, and `0.0200` MAE zero-shot on $n=128$ (28 Bitonic stages, 0 NaNs!).
4. **Degree vs. Scale-Stability Trade-Off:**
   - Increasing block degree from cubic ($m=1$, deg 27) to quintic ($m=2$, deg 125) cuts in-domain Value MAE by **$4.5\times$** (`0.0333 -> 0.0072`), while tightening the upper scale stability radius from $R_{\text{stable}} = 1.3$ down to $R_{\text{stable}} = 1.0$.
