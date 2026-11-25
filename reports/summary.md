# Executive Scientific Summary: Strictly Pure Polynomial Neural Networks for Sorting

**Author:** Libor "Burny" Burian  
**Date:** 2026-11-25  
**Paper:** [reports/paper.pdf](paper.pdf) | **Live Browser Sorter:** [https://burnycoder.github.io/polynomial-neural-network-sort-list/](https://burnycoder.github.io/polynomial-neural-network-sort-list/) | **Checkpoints:** [Hugging Face Hub](https://huggingface.co/BurnyCoder/polynomial-neural-network-sort-list)

---

## 1. Research Question & Core Takeaway
We asked whether a **strictly pure Polynomial Neural Network (PNN)**—using *exclusively* additions, subtractions, and multiplications at inference (mechanically verified via PyTorch FX with zero `ReLU`, `sigmoid`, `tanh`, `softmax`, `div`, `abs`, `min`/`max`, or normalization)—can learn to sort continuous real-valued lists, how architectural inductive bias governs accuracy and generalization, and whether gradient-trained composite polynomials beat hand-derived CKKS homomorphic-encryption minimax polynomials at identical polynomial degree.

Across **178 preregistered experimental runs** (17 architectures, 5 independent seeds on all 10 core models $\times$ $n \in \{8, 16\}$, 10 OOD suites, 6 causal ablations, and `float64`/hyperparameter audits), the answer is a **nuanced, causally verified yes — with sharp algebraic boundaries**:

### Most Significant Positive Findings
1. **36–60 parameter pure polynomials crush unstructured PNNs and beat analytical CKKS by $1.6\times\text{--}3.9\times$:**
   - `PolySortNet-Adaptive (m=2)` (deg 125/gate) achieves 5-seed held-out MAE **`0.00724 ± 0.00000`** ($n=8$, `95.9%` Ex@0.05) and **`0.00981 ± 0.00008`** ($n=16$, `81.8%` Ex@0.05) — beating `DeepSets-Poly` (131k params) by **$4.56\times / 4.20\times$**, `PiNet-NCP-Skip` (202k params) by **$7.49\times / 6.35\times$**, `PolyMLP` by **$8.26\times / 6.61\times$**, non-polynomial `ReLU/GELU-MLP` by **$4.90\times / 6.48\times$**, and analytical CKKS Cheon/Hong/Mazzone $f_3\circ g_3$ by **$3.85\times$** (all Holm-Bonferroni paired Wilcoxon $p_{\text{holm}} < 10^{-300}$, 5/5 seeds).
2. **Zero-shot $16\times$ length extrapolation & OOD wins over 96.6k-parameter `SetTransformer`:**
   - 6-parameter `PolySortNet-Shared (m=2)` trained only on $n=8$ extrapolates zero-shot to **$n=128$** (28 Bitonic stages) with **MAE `0.02001` ($\tau = 0.9978$)**, beating `SetTransformer` (`0.05311`) by **$2.65\times$**.
   - On 5-seed $n=8$ averages, 36-param `PolySortNet-Adaptive (m=2)` beats 96.6k-param `SetTransformer` on **Near-Ties $10^{-3}$** (`0.00659` vs `0.00665`), **50% Duplicates** (`0.00660` vs `0.00737`), and **Shift $+100$** (`0.00726` vs `99.50`, **$13{,}714\times$**), with exact float64 sum conservation ($2.1 \times 10^{-16}$) and shift equivariance ($6.6 \times 10^{-14}$).
3. **Iso-parameter depth separation & autonomous Bitonic stride rediscovery:**
   - Holding parameter count fixed at 4/stage, composite `L=4, d=3` (deg 81) beats flat `L=1, d=9` by **$4.28\times / 4.13\times$** (`0.0132 / 0.0163`, `Ex@0.05 = 91.8% / 61.4%`) while expanding upper scale stability from `0.00` (Runge blow-up) $\to$ **`1.50`**.
   - Stage-adaptive slopes $s'(0)$ autonomously dip at every long-stride Bitonic merge reset (Stage 3 for $n=8$; Stages 1, 3, 6 for $n=16$) and ramp monotonically as stride halves.
4. **Orders-of-magnitude lower seed variance & zero `float32` drift:**
   - `PolySortNet-Adaptive (m=2)` exhibits seed SD `5.8e-7` at $n=8$ ($9{,}078\times$ smaller than `PiNet`); full `float64` training matches `float32` to 6 decimal places even at degree 81.

### Most Significant Negative Results & Falsified Preregistered Hypotheses
1. **H6 Part 1 FALSIFIED on ID MAE:** Removing the $h(1)=1$ fixed-point constraint (`unconstrained_odd_cheby`, `raw_monomial`, `general_cheby`) **improves** ID MAE by 26–30% (`0.0234 / 0.0273` vs `0.0333 / 0.0369`) by contracting late-stage comparator domains ($s_5(1) = -445 \text{ to } -1038$), trading away scale stability ($R_{\text{stable}}: 1.3 \to 1.1$).
2. **H3 Part 1 FALSIFIED at strict iso-degree $D=9$:** 4-param/stage flat `L=1, d=9` (`0.0566 / 0.0674`) beats 2-param/stage composite `L=2, d=3` (`0.0771 / 0.0831`) on ID MAE before suffering Runge blow-up at $1.1\times$ scale.
3. **Inevitable pure-polynomial upper scale cliff (H5 causally proven):** Every pure polynomial blows up at $s = 1.5\text{--}2.0$; non-poly `max(|x|)` wrapping (A6) immediately restores $R_{\text{stable}} = 5.0$ while breaking Shift $+100$ equivariance (`0.455 / 0.480`).
4. **Non-polynomial upper bounds still lead in-distribution:** Transcendental `DiffSort-Logistic` (`0.00041 / 0.00069`), `DiffSort-Cauchy` (`0.00306 / 0.00424`), and `SetTransformer` (`0.00218 / 0.00377`) remain sharper on bounded ID lists.
5. **Unstructured PNNs collapse at $n=16$ & H1 cubic caveat:** `PiNet`, `PolyMLP`, `ReLU-MLP`, and `PolyPairwiseRank` all score `0.0%` Ex@0.05 at $n=16$; cubic $m=1$ `PolySortNet` ties `DeepSets-Poly` at $n=8$ (`0.03327` vs `0.03300`) before dominating at $n=16$ and on OOD shifts/duplicates.
