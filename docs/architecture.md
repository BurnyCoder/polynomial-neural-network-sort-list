# Architecture Documentation (`pnn-sort`)

This document describes the modular software and mathematical architecture of `pnn-sort`.

## 1. Software Pipeline Flow

```text
src/pnn_sort/pipeline.py   (Top-level readable orchestrator)
 ├── data/                 Deterministic SHA-256 keyed train/val/test + 10 OOD stress suites
 ├── models/               Strictly pure PNNs, analytical CKKS models, controls, non-poly baselines
 ├── audit.py              Automated PyTorch FX pure-polynomial verifier + degree/depth analyzer
 ├── train/                Resumable AdamW/cosine trainer, losses, atomic checkpoints, JSONL logger
 ├── dashboard/            Real-time local HTTP live metrics monitor during training
 ├── eval/                 Metrics (MAE, Kendall tau, monotonicity, W1, sum error), 95% CIs, Wilcoxon
 ├── viz/                  Publication-grade figures, architecture diagrams, output examples
 └── publish/              GitHub & Hugging Face Hub sync + browser weight export
```

## 2. Core Mathematical Architectures

### 2.1 Fixed-Point Odd Chebyshev Composite Polynomial (`CompositeChebyshevSign`)
For scaled input $u \in [-1, 1]$, we compose $L$ odd residual Chebyshev blocks:
$$h_k(u) = u + u(1 - u^2) \sum_{j=0}^{m-1} c_{k,j}\, T_{2j}(u), \qquad s_\theta(u) = (h_L \circ \dots \circ h_1)(u),$$
using strictly polynomial Chebyshev recurrences ($T_0 = 1, T_2 = 2u^2 - 1, T_4 = 8u^4 - 8u^2 + 1, \dots$).
- Degree per block: $2m + 1$.
- Total degree of $s_\theta$: $(2m + 1)^L$.
- Guaranteed algebraic invariants for all $\theta$: $s_\theta(-u) = -s_\theta(u)$, $s_\theta(0) = 0$, $s_\theta(\pm 1) = \pm 1$.

### 2.2 Pure-Polynomial Sorting Network (`PolySortNet`)
Uses constant binary/half-weight sorting-network routing matrices $(S_a^{(t)}, S_b^{(t)}, C_{\min}^{(t)}, C_{\max}^{(t)})$ for Bitonic (depth $\frac{1}{2}\log_2 n(\log_2 n + 1)$) or Odd-Even networks (adapted directly from `Felix-Petersen/diffsort`). At each layer $t$:
$$a = x S_a^{(t)T}, \quad b = x S_b^{(t)T}, \quad \Delta = \frac{b - a}{2}, \quad \delta_{\text{abs}} = \Delta \cdot s_\theta^{(t)}(\Delta),$$
$$\hat{\min} = \frac{a + b}{2} - \delta_{\text{abs}}, \quad \hat{\max} = \frac{a + b}{2} + \delta_{\text{abs}}, \quad x \leftarrow \hat{\min}\, C_{\min}^{(t)T} + \hat{\max}\, C_{\max}^{(t)T}.$$
- **Shared-Gate (`polysort_shared`):** All layers share one $s_\theta$, enabling **zero-shot list-length generalization** to arbitrary $n$.
- **Stage-Adaptive (`polysort_adaptive`):** Each layer $t = 1, \dots, \ell$ learns independent coefficients tailored to its intermediate gap distribution.

### 2.3 Constant-Depth All-to-All Polynomial Ranker (`PolyPairwiseRank`)
Implements Mazzone et al. (USENIX Security 2025, [arXiv:2412.15126](https://arxiv.org/abs/2412.15126)) with learned composite Chebyshev polynomials:
1. $C_{ij} = \frac{1 + s_\theta^{\text{cmp}}((x_j - x_i)/2)}{2}$ (all pairs in parallel).
2. Exact quadratic tie-correction $E_{ij} = 4 C_{ij}(1 - C_{ij})$, $r_j = \sum_i C_{ij} + \sum_{i \le j} E_{ij} - \frac{1}{2}\sum_i E_{ij}$.
3. Polynomial rank indicator $W_{jk} = \frac{1 + s_\theta^{\text{ind}}((r_j - k + 0.5)/n)}{2} \cdot \frac{1 - s_\theta^{\text{ind}}((r_j - k - 0.5)/n)}{2}$, output $\hat{y}_k = \sum_j W_{jk} x_j$.

### 2.4 Unstructured & Symmetric Pure PNNs
- **`PiNetSorter` (Chrysos et al., TPAMI 2021, Eq. 10 & Sec. 3.2):** Pure ProdPoly NCP-Skip without BatchNorm or `tanh`.
- **`DeepSetsPoly` (Zaheer et al., 2017; Wagstaff et al., 2022):** Permutation-invariant polynomial power/Chebyshev sum pooling ($\text{latent width} \ge 2n$) decoded by a pure ProdPoly NCP-Skip head.
- **`PolyMLP` (Kileel et al., 2019):** Dense layers with learnable degree-3 Chebyshev activations.
