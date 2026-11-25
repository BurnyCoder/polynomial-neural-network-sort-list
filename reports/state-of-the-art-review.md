# State-of-the-Art Review: Polynomial & Neural Methods for Sorting Lists

**Author:** Libor "Burny" Burian  
**Date:** 2026-11-25  

---

## 1. Taxonomy of State-of-the-Art Approaches

No prior published work systematically trains and compares **strictly pure Polynomial Neural Networks (PNNs)** across architectural inductive biases for continuous numerical list sorting. Instead, the state of the art spans **four distinct paradigms**, summarized below.

| Paradigm | Representative Methods | Pure Polynomial? | Comparison / Layer Depth | Zero-Shot Length Generalization? | Handles Exact Ties? | Primary Strength | Key Limitation for Pure-PNN List Sorting |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **1. Unstructured Deep PNNs** | $\Pi$-Nets (CCP / NCP / NCP-Skip / ProdPoly; Chrysos et al., CVPR '20 / TPAMI '21), PolyMLP (Kileel et al., NeurIPS '19) | Yes *(when BN/tanh removed)* | $O(L)$ | No (fixed $n$) | Approximate | General multivariate polynomial approximator; tensor-factorized parameters | Lacks permutation symmetry & pairwise comparison bias; severe gradient instability & Bernstein $O(1/D)$ bottleneck |
| **2. Symmetric Set Networks** | DeepSets (Zaheer et al., NeurIPS '17), Power-Sum PNN, Set Transformer (Lee et al., ICML '19) | DeepSets-Poly: Yes; SetTransf: No | $O(L)$ | SetTransf: Yes; DeepSets: latent $\ge n$ req. | Approximate | Enforces exact permutation invariance $f(\pi x)=f(x)$ | Wagstaff et al. (2019/2022) prove latent width $\ge n$ is mandatory; decoding order statistics from power sums is ill-conditioned |
| **3. Differentiable Sorting Networks (Non-Poly)** | `diffsort` Logistic / ART / Cauchy / Optimal (Petersen et al., ICML '21 & ICLR '22); NeuralSort (ICLR '19); SoftSort (ICML '20); FastSort (ICML '20) | **No** ($\exp$, $\arctan$, division, `abs`) | Bitonic: $\frac{1}{2}\log_2 n(\log_2 n + 1)$; Odd-Even: $n$ | Yes | Yes (exact on CAS wires) | Proven monotonicity + $\epsilon \ell$ error bound (Petersen Thm. 5, 8, 9); scales to $n=1024$ | Uses non-polynomial transcendental activations; temperature $\beta$ tuned via grid search rather than learned per-stage polynomials |
| **4. Analytical Polynomial Sorting (CKKS FHE)** | Cheon et al. (ASIACRYPT '20), Hong et al. (IEEE TIFS '21), Lee et al. (IEEE TDSC '22), Mazzone et al. (USENIX Sec '25) | **Yes** (additions & multiplications only) | Bitonic: $O(\log^2 n)$; $k$-way: $O(k\log_k^2 n)$; Mazzone: **$O(1)$ (depth 2)** | Yes | Swap nets: Yes; Mazzone: Yes w/ $E=4C(1-C)$ | Optimal asymptotic depth for worst-case $L_\infty$ sign error on $[\delta, 1]$ | **Zero data adaptation:** fixed coefficients optimize unweighted $\|1 - s(u)\|_\infty$ rather than value-sorting error $u|1-s(u)|$ or stage-varying gap distributions |

---

## 2. Deep Technical Analysis of SOTA Building Blocks

### 2.1 Why Analytical Sign-Minimax Polynomials Are Suboptimal for Value Sorting
In a polynomial compare-and-swap (CAS) gate with scaled difference $\Delta = \frac{b - a}{2} \in [-1, 1]$ and odd polynomial $s(\Delta) \approx \mathrm{sign}(\Delta)$:
$$\hat{\min}_s(a, b) = \frac{a+b}{2} - \Delta \cdot s(\Delta), \qquad \hat{\max}_s(a, b) = \frac{a+b}{2} + \Delta \cdot s(\Delta).$$
The exact single-gate **value error** is:
$$e_{\text{val}}(\Delta) = \big|\hat{\min}_s(a,b) - \min(a,b)\big| = \big|\hat{\max}_s(a,b) - \max(a,b)\big| = |\Delta| \cdot \big|1 - s(|\Delta|)\big|.$$

Compare the two optimization objectives on $[0, 1]$:
1. **Analytical CKKS Objective (Cheon et al., 2020; Lee et al., 2022):** Minimizes unweighted sign error $\sup_{u \in [\delta, 1]} |1 - s(u)|$ for a chosen precision threshold $\delta > 0$. Because $s(0) = 0$ is fixed, forcing small error at tiny $\delta$ forces high slope near $0$, causing Gibbs-like oscillations or requiring high multiplicative depth $L$.
2. **Value-Sorting Objective (This Study):** Minimizes expected value error $\mathbb{E}\big[\|\hat{y} - \mathrm{sort}(x)\|_p^p\big]$ over the actual input and intermediate-layer gap distributions! Crucially:
   - The factor $|\Delta|$ naturally **down-weights** tiny gaps ($|\Delta| \to 0$) in value error, allowing a low-depth polynomial to achieve far lower MAE/RMSE than an unweighted sign-minimax polynomial of the *exact same degree and multiplicative depth*.
   - **Stage-Dependent Gap Distribution:** In an $\ell$-layer Bitonic sorting network (e.g., $\ell = 6$ for $n=8$, $\ell = 10$ for $n=16$), early layers compare nearly random pairs with wide gaps ($|\Delta| \sim O(1)$), whereas final merge layers compare adjacent sorted sub-lists with much smaller gaps ($|\Delta| \sim O(1/n)$). Analytical CKKS networks apply the **identical** polynomial $s$ at every stage, wasting depth early or under-resolving late; **stage-adaptive learned `PolySortNet`** can allocate distinct composite polynomial coefficients $\{c^{(t)}_{k,j}\}$ to each sorting layer $t = 1, \dots, \ell$!

### 2.2 Fixed-Point-Preserving Chebyshev Residual Parameterization
Training deep compositions of raw monomials $s(u) = \sum c_j u^j$ is notoriously ill-conditioned (Vandermonde condition numbers grow exponentially in degree, and small parameter steps cause activations to blow up beyond $[-1, 1]$).

To solve this cleanly while remaining **100% strictly polynomial**, we introduce the **Fixed-Point Odd Chebyshev Composition Block**:
$$h_k(u) = u + u(1 - u^2) \sum_{j=0}^{m-1} c_{k,j}\, T_{2j}(u), \qquad s_\theta(u) = (h_L \circ h_{L-1} \circ \dots \circ h_1)(u),$$
where $T_0(u) = 1$, $T_2(u) = 2u^2 - 1$, $T_4(u) = 8u^4 - 8u^2 + 1$, etc., computed strictly via polynomial recurrence $T_{r+1}(u) = 2u T_r(u) - T_{r-1}(u)$.

**Exact Mathematical Properties:**
1. **Strictly Pure Polynomial:** Degree of each block is $2m + 1$; total degree of $s_\theta$ is $(2m+1)^L$; total degree of the CAS gate $\Delta s_\theta(\Delta)$ is $(2m+1)^L + 1$.
2. **Exact Odd Symmetry:** $h_k(-u) = -h_k(u) \implies s_\theta(-u) = -s_\theta(u)$ identically for all $\theta$.
3. **Exact Fixed Points:** $h_k(0) = 0$ and $h_k(\pm 1) = \pm 1$ identically for all $\theta$.
4. **Exact Subsumption of Analytical SOTA:**
   - $c_{k,j} = 0 \implies h_k(u) = u$ (exact identity — ideal residual initialization).
   - $m = 1, c_{k,0} = \frac{1}{2} \implies h_k(u) = \frac{3}{2}u - \frac{1}{2}u^3 = f_1(u)$ (Cheon et al., 2020).
   - $m = 2, c_{k,0} = \frac{11}{16}, c_{k,1} = -\frac{3}{16} \implies$ wait, let's verify exact Chebyshev expansion of $f_2(u) = u + u(1-u^2)(\frac{1}{2} + \frac{3}{8}(1-u^2)) = u + u(1-u^2)(\frac{7}{8} - \frac{3}{8}u^2)$! Since $u^2 = \frac{1 + T_2(u)}{2}$, we have $\frac{7}{8} - \frac{3}{8}\cdot\frac{1+T_2(u)}{2} = \frac{11}{16} - \frac{3}{16}T_2(u)$! Exact algebraic identity!!
5. **Bounded Propagation Guarantee:** Whenever $|s_\theta(\Delta)| \le 1$ on $[-1, 1]$, every wire at every depth remains strictly inside $[\min(x), \max(x)] \subseteq [-1, 1]$, and Petersen et al. (2022) Theorem 9 gives the exact $L_\infty$ upper bound:
   $$\|\hat{y} - \mathrm{sort}(x)\|_\infty \le \ell \cdot \sup_{u \in [0, 1]} u\, |1 - s_\theta(u)|.$$

### 2.3 Constant-Depth All-to-All Learned Polynomial Sorting (`PolyPairwiseRank`)
Inspired directly by **Mazzone et al. (USENIX Security 2025, arXiv:2412.15126)**, we also design a **learned constant-comparison-depth** pure polynomial sorter that avoids the $O(\log^2 n)$ sequential CAS depth of Bitonic networks:
1. Compute all $n \times n$ pairwise scaled differences $\Delta_{ij} = (x_j - x_i)/2$ and learned odd composite polynomial comparison matrix $C_{ij} = \frac{1 + s_\theta^{\text{cmp}}(\Delta_{ij})}{2}$.
2. Apply Mazzone's exact polynomial tie-correction offset (Sec. 4, Alg. 6):
   $$E_{ij} = 4 C_{ij}(1 - C_{ij}), \qquad r_j = \sum_{i=1}^n C_{ij} + \sum_{i \le j} E_{ij} - \frac{1}{2}\sum_{i=1}^n E_{ij} \in [1, n].$$
3. For each target rank $k \in \{1, \dots, n\}$, apply a learned polynomial indicator kernel centered at $k$ using normalized rank difference $z_{jk} = \frac{r_j - k}{n} \in [-1, 1]$:
   $$W_{jk} = \frac{1 + s_\theta^{\text{ind}}\big(z_{jk} + \frac{1}{2n}\big)}{2} \cdot \frac{1 - s_\theta^{\text{ind}}\big(z_{jk} - \frac{1}{2n}\big)}{2},$$
   and output $\hat{y}_k = \sum_{j=1}^n W_{jk} x_j$.
- **Trade-off to test empirically:** `PolyPairwiseRank` has **constant comparison depth** (depth 2 instead of $\frac{1}{2}\log_2 n(\log_2 n + 1)$), is 100% pure polynomial, and generalizes zero-shot to arbitrary $n$ — **but** its indicator step must resolve rank spacing $1/n$ (shrinking as $n$ grows) AND lacks exact sum-conservation unless $\sum_k W_{jk} = 1$. Comparing `PolySortNet` vs. `PolyPairwiseRank` directly tests the **swap-network vs. rank-indicator** paradigm divide in trainable PNNs!

---

## 3. Summary of Key Trade-offs & Open Questions Resolved by Our Experiments

| Scientific Question | Analytical / Theoretical Expectation | Empirical Test Designed in This Study |
| :--- | :--- | :--- |
| **Can unstructured PNNs ($\Pi$-Net, PolyMLP) sort lists?** | Expressible via Stone-Weierstrass, but suffers Bernstein $O(1/D)$ bottleneck & combinatorial lack of permutation bias; should degrade rapidly as $n$ grows from $4 \to 16$. | Compare `pinet_ncp_skip`, `poly_mlp`, `deepsets_poly` vs `polysort_*` across $n \in \{4, 8, 16\}$. |
| **Can learned PNNs beat analytical CKKS minimax at iso-depth?** | **Yes for Value MAE/RMSE** (value error weights sign error by $|\Delta|$, and stage-adaptive polynomials exploit narrowing layer gaps); **trade-off on adversarial near-ties** unless ordering margin loss is included. | Iso-depth head-to-head: `polysort_shared` & `polysort_adaptive` vs `analytical_cheon` & `analytical_remez` on Uniform ID + Adversarial Near-Ties. |
| **Is flat-vs-composite depth separation empirically large?** | Huge separation predicted: degree-9 composite ($L=2, d=3$) should vastly outperform flat degree-9 ($L=1, d=9$) in optimization stability and accuracy, and $L=3$ ($d=3$, deg 27) should improve further. | Iso-degree ablation: $(L=1, d=9)$ vs $(L=2, d=3)$ vs $(L=3, d=3)$ vs $(L=4, d=3)$. |
| **Can sorting wiring be learned end-to-end vs fixed Bitonic?** | Unconstrained linear mixing breaks exact sum conservation & swap structure; likely worse than Bitonic/Odd-Even, while still beating random fixed wiring. | `polysort_learned_wire` vs `polysort_shared` (Bitonic/Odd-Even) vs `random_wire` negative control. |
| **Where do pure polynomials fundamentally fail?** | Exact polynomial divergence beyond stability radius $R$ on scaled OOD inputs ($s > 1$). | Scale extrapolation sweep $s \in [0.1, 5.0]$ + non-poly scale-wrapper ablation. |

---

## References (Verified Primary Sources)

1. Batcher, K. E. (1968). *Sorting networks and their applications*. AFIPS Spring Joint Computer Conference, pp. 307–314.
2. Bernstein, S. N. (1914). *Sur la meilleure approximation de $|x|$ par des polynômes de degrés donnés*. Acta Mathematica, 37, 1–57.
3. Blondel, M., Teboul, O., Berthet, Q., & Djolonga, J. (2020). *Fast Differentiable Sorting and Ranking*. ICML 2020. [arXiv:2002.08871](https://arxiv.org/abs/2002.08871).
4. Cheon, J. H., Kim, D., & Kim, D. (2020). *Efficient Homomorphic Comparison Methods with Optimal Complexity*. ASIACRYPT 2020. [IACR ePrint 2019/1234](https://eprint.iacr.org/2019/1234).
5. Chrysos, G. G., Moschoglou, S., Bouritsas, G., Panagakis, Y., Deng, J., & Zafeiriou, S. (2020). *$\Pi$-nets: Deep Polynomial Neural Networks*. CVPR 2020. [arXiv:2003.03828](https://arxiv.org/abs/2003.03828).
6. Chrysos, G. G., Moschoglou, S., Bouritsas, G., Deng, J., Panagakis, Y., & Zafeiriou, S. (2021). *Deep Polynomial Neural Networks*. IEEE TPAMI. [arXiv:2006.13026](https://arxiv.org/abs/2006.13026) | [GitHub](https://github.com/grigorisg9gr/polynomial_nets).
7. Cuturi, M., Teboul, O., & Vert, J.-P. (2019). *Differentiable Ranks and Sorting using Optimal Transport*. NeurIPS 2019. [arXiv:1905.11885](https://arxiv.org/abs/1905.11885).
8. Grover, A., Wang, E., Zweig, A., & Ermon, S. (2019). *Stochastic Optimization of Sorting Networks via Continuous Relaxations (NeuralSort)*. ICLR 2019. [arXiv:1903.08850](https://arxiv.org/abs/1903.08850).
9. Hong, S., Kim, S., Choi, J., Lee, Y., & Cheon, J. H. (2021). *Efficient Sorting of Homomorphic Encrypted Data with $k$-Way Sorting Network*. IEEE TIFS, 16, 4389–4404.
10. Ivakhnenko, A. G. (1971). *Polynomial Theory of Complex Systems*. IEEE Trans. Systems, Man, and Cybernetics, (4), 364–378.
11. Kileel, J., Trager, M., & Bruna, J. (2019). *On the Expressive Power of Deep Polynomial Neural Networks*. NeurIPS 2019. [arXiv:1905.12207](https://arxiv.org/abs/1905.12207).
12. Lee, E., Lee, J.-W., No, J.-S., & Kim, Y.-S. (2022). *Minimax Approximation of Sign Function by Composite Polynomial for Homomorphic Comparison*. IEEE TDSC, 19(6), 3711–3727. [IACR ePrint 2020/834](https://eprint.iacr.org/2020/834).
13. Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019). *Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks*. ICML 2019. [arXiv:1810.00825](https://arxiv.org/abs/1810.00825).
14. Lipton, Z. C., & Steinhardt, J. (2018). *Troubling Trends in Machine Learning Scholarship*. [arXiv:1807.03341](https://arxiv.org/abs/1807.03341).
15. Lones, M. A. (2021). *How to Avoid Machine Learning Pitfalls: A Guide for Academic Researchers*. [arXiv:2108.02497](https://arxiv.org/abs/2108.02497).
16. Madsen, A., & Johansen, A. R. (2020). *Neural Arithmetic Units*. ICLR 2020. [arXiv:2001.05016](https://arxiv.org/abs/2001.05016).
17. Mazzone, F., Everts, M., Hahn, F., & Peter, A. (2025). *Efficient Ranking, Order Statistics, and Sorting under CKKS*. USENIX Security '25. [arXiv:2412.15126](https://arxiv.org/abs/2412.15126).
18. Petersen, F., Borgelt, C., Kuehne, H., & Deussen, O. (2021). *Differentiable Sorting Networks for Scalable Sorting and Ranking Supervision*. ICML 2021, PMLR 139:8546–8555. [arXiv:2105.04019](https://arxiv.org/abs/2105.04019) | [GitHub](https://github.com/Felix-Petersen/diffsort).
19. Petersen, F., Borgelt, C., Kuehne, H., & Deussen, O. (2022). *Monotonic Differentiable Sorting Networks*. ICLR 2022. [arXiv:2203.09630](https://arxiv.org/abs/2203.09630).
20. Prillo, S., & Eisenschlos, J. M. (2020). *SoftSort: A Continuous Relaxation for the argsort Operator*. ICML 2020. [arXiv:2006.16038](https://arxiv.org/abs/2006.16038).
21. Shin, Y., & Ghosh, J. (1991). *The Pi-Sigma Network: An Efficient Higher-Order Neural Network for Pattern Classification and Function Approximation*. IJCNN 1991, vol. 1, pp. 13–18.
22. Sidharth, S. S., Keerthana, A. R., Gokul, R., & Anas, K. P. (2024). *Chebyshev Polynomial-Based Kolmogorov-Arnold Networks (ChebyKAN)*. [arXiv:2405.07200](https://arxiv.org/abs/2405.07200).
23. Trask, A., Hill, F., Reed, S. E., Rae, J., Dyer, C., & Blunsom, P. (2018). *Neural Arithmetic Logic Units*. NeurIPS 2018. [arXiv:1808.00508](https://arxiv.org/abs/1808.00508).
24. Varga, R. S., & Carpenter, A. J. (1985/1987). *On the Bernstein Conjecture in Approximation Theory*. Numerische Mathematik 47, 179–196; Constructive Approximation 3, 193–223.
25. Vinyals, O., Fortunato, M., & Jaitly, N. (2015). *Pointer Networks*. NeurIPS 2015. [arXiv:1506.03134](https://arxiv.org/abs/1506.03134).
26. Wagstaff, E., Fuchs, F. B., Engelcke, M., Posner, I., & Osborne, M. A. (2019/2022). *On the Limitations of Representing Functions on Sets* (ICML 2019, [arXiv:1901.09006](https://arxiv.org/abs/1901.09006)) & *Universal Approximation of Functions on Sets* (JMLR 23(151):1–56, 2022).
27. Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R., & Smola, A. (2017). *Deep Sets*. NeurIPS 2017. [arXiv:1703.06114](https://arxiv.org/abs/1703.06114).
