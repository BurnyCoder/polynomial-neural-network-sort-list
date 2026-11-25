# Literature Review: Polynomial Neural Networks and Neural List Sorting

**Author:** Libor "Burny" Burian  
**Date:** 2026-11-25  
**Scope:** Strictly pure Polynomial Neural Networks (PNNs), polynomial approximation of piecewise-linear order statistics, symmetric set representations, differentiable sorting networks, and composite polynomial sorting under approximate Homomorphic Encryption (CKKS).

> **Verification Note:** All arXiv metadata and full-text primary sources cited below were retrieved and audited via the arXiv API (`https://info.arxiv.org/help/api/index.html`) and official public GitHub repositories (`grigorisg9gr/polynomial_nets`, `Felix-Petersen/diffsort`). Individual paper licenses (e.g., CC BY 4.0 for arXiv:2412.15126) apply to their respective source works.

---

## 1. Mathematical Nature of the Sorting Map

Let $x = (x_1, \dots, x_n) \in [-1, 1]^n$. The value-sorting map
$$S(x) = \big(x_{(1)}, x_{(2)}, \dots, x_{(n)}\big), \qquad x_{(1)} \le x_{(2)} \le \dots \le x_{(n)}$$
has five structural properties that directly constrain any polynomial approximator:

1. **Permutation Invariance (Symmetry):** For every permutation $\pi \in \mathfrak{S}_n$, $S(\pi x) = S(x)$. Each coordinate $x_{(k)}$ is a symmetric function of $(x_1, \dots, x_n)$.
2. **Continuous, Piecewise-Linear, and Non-Smooth:** $S$ is globally $1$-Lipschitz in $\|\cdot\|_\infty$ and $\|\cdot\|_2$, yet non-differentiable along all $\binom{n}{2}$ hyperplanes $x_i = x_j$. For $n = 2$,
   $$\min(a, b) = \frac{a + b - |a - b|}{2}, \qquad \max(a, b) = \frac{a + b + |a - b|}{2}.$$
   Because every finite-degree polynomial is $C^\infty$ analytic, **no finite-degree polynomial can represent $S(x)$ identically** on any open set intersecting $x_i = x_j$.
3. **Shift Equivariance & Sum Conservation:** $S(x + c\mathbf{1}) = S(x) + c\mathbf{1}$ for all $c \in \mathbb{R}$, and $\sum_{k=1}^n x_{(k)} = \sum_{i=1}^n x_i$.
4. **Positive Homogeneity (Scale Equivariance):** $S(\lambda x) = \lambda S(x)$ for $\lambda \ge 0$, and negation anti-symmetry $S(-x) = -\mathrm{rev}(S(x))$.
5. **Fundamental Polynomial Limitation on Scale Extrapolation:** Any non-affine polynomial $p : \mathbb{R}^n \to \mathbb{R}^n$ of degree $D \ge 2$ grows asymptotically as $O(\|x\|^D)$ along non-vanishing leading directions. Therefore, **strictly pure polynomials cannot maintain bounded error on unbounded domains** — they necessarily possess a finite stability radius $R$ beyond which $|x_i| > R$ diverges polynomially.

### 1.1 Approximation Theory of $|x|$: Flat vs. Composite Polynomials
By the Stone-Weierstrass theorem (Stone, 1948), continuous functions on compact sets $[-1, 1]^n$ can be uniformly approximated arbitrarily closely by polynomials. However, the **degree cost** differs exponentially between flat and deep (composite) polynomials:

- **Bernstein's Theorem (1914) and Varga & Carpenter (1985, 1987):** Let $E_D(|u|) = \inf_{\deg(p) \le D} \sup_{u \in [-1,1]} \big||u| - p(u)\big|$. Bernstein (1914) proved that $\beta = \lim_{m \to \infty} 2m\, E_{2m}(|u|)$ exists. Using high-precision Remez exchange computations, Varga & Carpenter (1985, *Numerische Mathematik* 47, 179–196; 1987, *Constructive Approximation* 3, 193–223) disproved Bernstein's conjecture ($\beta = \frac{1}{2\sqrt{\pi}}$) and established:
  $$\beta \approx 0.2801694990, \qquad E_D(|u|) \sim \frac{\beta}{D}.$$
  Consequently, a **flat polynomial** requires degree $D = \Theta(1/\varepsilon)$ to approximate a single two-element comparator to error $\varepsilon$.
- **Exponential Degree Compression via Composition:** Nesting $L$ low-degree odd polynomials $s(u) = (h_L \circ \dots \circ h_1)(u)$ of degree $d$ attains total polynomial degree $D = d^L$ with only $O(L)$ parameters and $O(L)$ multiplications, driving error down super-exponentially on $[-1, -\delta] \cup [\delta, 1]$ (Cheon et al., 2020; Lee et al., 2022).

---

## 2. Polynomial Neural Networks (PNNs)

### 2.1 Classical Polynomial Networks
- **GMDH (Ivakhnenko, 1971):** *Polynomial Theory of Complex Systems*, IEEE Trans. Systems, Man, and Cybernetics, no. 4, pp. 364–378. Introduced hierarchical quadratic partial descriptors on predefined variable pairs (often cited as the first deep polynomial network; cf. Chrysos et al., 2021, Sec. 2).
- **Pi-Sigma Networks (Shin & Ghosh, 1991):** *The Pi-Sigma Network: An Efficient Higher-Order Neural Network for Pattern Classification and Function Approximation*, IJCNN 1991, vol. 1, pp. 13–18. Sums linear projections then multiplies them via a product unit; extended to Sigma-Pi-Sigma networks (SPSNN; Li, 2003, *Neural Processing Letters*). Struggles to scale to high-dimensional or high-degree settings without factorization.

### 2.2 Expressive Power of Deep Polynomial Neural Networks
- **Kileel, Trager & Bruna (NeurIPS 2019):** *On the Expressive Power of Deep Polynomial Neural Networks* ([arXiv:1905.12207](https://arxiv.org/abs/1905.12207)). Analyzes feedforward networks with pure polynomial activations $x \mapsto x^r$ through algebraic geometry, defining the **functional variety** (Zariski closure of representable polynomials). Key findings:
  - Networks with narrow intermediate widths suffer a **bottleneck property** preventing them from filling the ambient polynomial space regardless of depth/other widths.
  - Architectures whose widths increase monotonically then decrease monotonically optimize variety dimension.

### 2.3 $\Pi$-Nets: Tensor-Factorized Deep Polynomial Neural Networks
- **Chrysos, Moschoglou, Bouritsas, Panagakis, Deng & Zafeiriou (CVPR 2020; IEEE TPAMI 2021):**
  - Conference version: *$\Pi$-nets: Deep Polynomial Neural Networks* ([arXiv:2003.03828](https://arxiv.org/abs/2003.03828)).
  - Journal version: *Deep Polynomial Neural Networks* ([arXiv:2006.13026](https://arxiv.org/abs/2006.13026), IEEE TPAMI; official code: [github.com/grigorisg9gr/polynomial_nets](https://github.com/grigorisg9gr/polynomial_nets)).
  - Follow-up: *Augmenting Deep Classifiers with Polynomial Neural Networks* (ECCV 2022, [arXiv:2104.07916](https://arxiv.org/abs/2104.07916)).
- **Exact Formulations (Chrysos et al., arXiv:2006.13026, Sec. 3.1–3.2, Eqs. 6, 9, 10):** To avoid the $O(d^N)$ tensor explosion of an $N$-th order multivariate polynomial $G(\mathbf{z}) = \sum_{n=1}^N \mathcal{W}^{[n]} \prod_{j=2}^{n+1} \times_j \mathbf{z} + \boldsymbol{\beta}$, Chrysos et al. introduce three coupled tensor factorizations implemented via Hadamard products ($*$):
  1. **CCP (Eq. 6):** $\mathbf{x}_n = (\mathbf{U}_{[n]}^T \mathbf{z}) * \mathbf{x}_{n-1} + \mathbf{x}_{n-1}$, with $\mathbf{x}_1 = \mathbf{U}_{[1]}^T \mathbf{z}$ and $\mathbf{x} = \mathbf{C}\mathbf{x}_N + \boldsymbol{\beta}$.
  2. **NCP (Eq. 9):** $\mathbf{x}_n = (\mathbf{A}_{[n]}^T \mathbf{z}) * (\mathbf{S}_{[n]}^T \mathbf{x}_{n-1} + \mathbf{B}_{[n]}^T \mathbf{b}_{[n]})$, with $\mathbf{x}_1 = (\mathbf{A}_{[1]}^T \mathbf{z}) * (\mathbf{B}_{[1]}^T \mathbf{b}_{[1]})$ and $\mathbf{x} = \mathbf{C}\mathbf{x}_N + \boldsymbol{\beta}$.
  3. **NCP-Skip (Eq. 10):** $\mathbf{x}_n = (\mathbf{A}_{[n]}^T \mathbf{z}) * (\mathbf{S}_{[n]}^T \mathbf{x}_{n-1} + \mathbf{B}_{[n]}^T \mathbf{b}_{[n]}) + \mathbf{V}_{[n]} \mathbf{x}_{n-1}$.
  4. **ProdPoly (Sec. 3.2):** Composing $M$ polynomial blocks of order $B$ produces polynomial degree $B^M$ with linear parameter count.
- **Crucial Caveat from Chrysos et al. (arXiv:2006.13026, Sec. 4–5):** In practice, Chrysos et al. frequently insert normalization (BatchNorm/InstanceNorm) or a final `tanh` to bound gradients, explicitly noting in Section 5 (*Future Directions*) that pure high-order polynomials without normalization can produce unbounded gradients and overfit. In our work, we test **strictly pure** $\Pi$-Nets (zero normalization/activation) as our unstructured PNN baseline.

### 2.4 Orthogonal Polynomial Networks & Arithmetic Neural Units
- **ChebyKAN (Sidharth et al., 2024):** *Chebyshev Polynomial-Based Kolmogorov-Arnold Networks: An Efficient Architecture for Nonlinear Function Approximation* ([arXiv:2405.07200](https://arxiv.org/abs/2405.07200)). Replaces B-splines in KANs (Liu et al., 2024, [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)) with Chebyshev polynomials $T_k(u)$ for orthogonal conditioning — note that standard ChebyKAN applies `tanh` before evaluating $T_k$ to clamp inputs to $[-1, 1]$, which breaks strict polynomiality! Our Chebyshev parameterization eliminates `tanh` by exploiting the algebraic range-boundedness of odd CAS gates on $[-1, 1]$.
- **NALU / NAU / NMU:** Trask et al. (NeurIPS 2018, *Neural Arithmetic Logic Units*, [arXiv:1808.00508](https://arxiv.org/abs/1808.00508)) and Madsen & Johansen (ICLR 2020, *Neural Arithmetic Units*, [arXiv:2001.05016](https://arxiv.org/abs/2001.05016)) highlight how carefully constrained parameterizations and initialization are essential for stable gradient learning of multiplicative/arithmetic circuits.

---

## 3. Symmetric Functions on Sets & Order Statistics

- **DeepSets (Zaheer et al., NeurIPS 2017):** *Deep Sets* ([arXiv:1703.06114](https://arxiv.org/abs/1703.06114)). Establishes that permutation-invariant set functions take the form $f(X) = \rho\big(\sum_{x \in X} \phi(x)\big)$. For polynomial $\phi(x) = (x, x^2, \dots, x^K)$ (or Chebyshev basis $T_k(x)$), $\sum_i \phi(x_i)$ computes the **Newton power sums**, which by the Fundamental Theorem of Symmetric Polynomials algebraically generate all symmetric polynomials in $(x_1, \dots, x_n)$.
- **Latent Dimension Bottleneck (Wagstaff et al., ICML 2019; JMLR 2022):** *On the Limitations of Representing Functions on Sets* ([arXiv:1901.09006](https://arxiv.org/abs/1901.09006)) and *Universal Approximation of Functions on Sets* (JMLR 23(151):1–56, 2022). Proves that for *continuous* encoder/decoder maps $\phi, \rho$, universal representation of set functions (explicitly including **order statistics / sorting**!) requires latent dimension $L_{\text{lat}} \ge n$. Consequently, our `DeepSetsPoly` baseline sets latent width $\ge 2n$.
- **Set Transformer (Lee et al., ICML 2019):** *Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks* ([arXiv:1810.00825](https://arxiv.org/abs/1810.00825)). Uses multi-head self-attention (softmax, non-polynomial) to capture pairwise interactions directly across arbitrary set sizes; serves as a strong non-polynomial neural baseline.
- **Pointer Networks (Vinyals et al., NeurIPS 2015):** *Pointer Networks* ([arXiv:1506.03134](https://arxiv.org/abs/1506.03134)). Early autoregressive attention baseline for combinatorial/sorting tasks (non-polynomial).

---

## 4. Differentiable Sorting & Sorting Networks

### 4.1 Classical Data-Oblivious Sorting Networks
- **Batcher (1968):** *Sorting Networks and Their Applications*, AFIPS Spring Joint Computer Conference, pp. 307–314. Introduced the **Bitonic** and **Odd-Even Mergesort** networks with parallel depth $\ell = \frac{1}{2}\log_2 n (\log_2 n + 1)$ and $O(n \log^2 n)$ compare-and-swap (CAS) gates.
- **Habermann (1972):** *Parallel Neighbor-Sort*, CMU Technical Report (odd-even transposition sorting network, depth $n$, $\approx n(n-1)/2$ CAS gates).
- **Knuth (1998):** *The Art of Computer Programming, Vol. 3: Sorting and Searching* (2nd ed.), Sec. 5.3.4 (zero-one principle and optimal-depth sorting networks).

### 4.2 Continuous Relaxations of Sorting (Non-Polynomial)
- **NeuralSort (Grover et al., ICLR 2019):** *Stochastic Optimization of Sorting Networks via Continuous Relaxations* ([arXiv:1903.08850](https://arxiv.org/abs/1903.08850)). Relaxes permutation matrices to unimodal row-stochastic matrices via temperature-controlled softmax over pairwise $|x_i - x_j|$ terms.
- **Optimal Transport / Sinkhorn Sorting (Cuturi et al., NeurIPS 2019):** *Differentiable Ranks and Sorting using Optimal Transport* ([arXiv:1905.11885](https://arxiv.org/abs/1905.11885)). Frames sorting as entropically regularized 1D optimal transport solved via iterative Sinkhorn projections.
- **Fast Differentiable Sorting and Ranking (Blondel et al., ICML 2020):** *Fast Differentiable Sorting and Ranking* ([arXiv:2002.08871](https://arxiv.org/abs/2002.08871)). Casts sorting as projection onto the permutahedron with $O(n \log n)$ isotonic regression; monotonic, but unbounded relaxation error (Petersen et al., 2022, Table 2).
- **SoftSort (Prillo & Eisenschlos, ICML 2020):** *SoftSort: A Continuous Relaxation for the argsort Operator* ([arXiv:2006.16038](https://arxiv.org/abs/2006.16038)). Streamlined $O(n^2)$ unimodal row-stochastic relaxation using row-wise softmax over $-|x_i - s_j|/\tau$.
- **Differentiable Sorting Networks (`diffsort`, Petersen et al., ICML 2021 & ICLR 2022):**
  - ICML 2021: *Differentiable Sorting Networks for Scalable Sorting and Ranking Supervision* ([arXiv:2105.04019](https://arxiv.org/abs/2105.04019), PMLR 139:8546–8555; code: [github.com/Felix-Petersen/diffsort](https://github.com/Felix-Petersen/diffsort)).
  - ICLR 2022: *Monotonic Differentiable Sorting Networks* ([arXiv:2203.09630](https://arxiv.org/abs/2203.09630)).
  - **Key Exact Results (verified from full PDF & `diffsort/functional.py`):**
    - Continuous CAS relaxation: $\min_f(a,b) = a f(b-a) + b f(a-b)$, $\max_f(a,b) = a f(a-b) + b f(b-a)$ with odd-symmetric $f(x) = 1 - f(-x)$ (guarantees symmetry, idempotency, shift invariance, sum conservation, and boundedness by hard $\min/\max$; Appendix B).
    - **Theorem 5 & 8 (ICLR 2022):** Global monotonicity requires $f'(x) \in \Omega(1/x^2)$, and bounded error requires $f'(x) \in O(1/x^2)$. Thus logistic $\sigma(\beta x)$ is error-bounded ($\approx 0.0696/\beta$) but non-monotonic; Cauchy CDF $f_{\mathcal{C}}(x) = \frac{1}{\pi}\arctan(\beta x) + \frac{1}{2}$ is $C^\infty$ analytic, monotonic, and error-bounded ($1/(\pi^2 \beta)$); optimal 1-Lipschitz monotonic $f_{\mathcal{O}}$ achieves error bound $1/(16\beta)$.
    - **Theorem 9 (ICLR 2022):** If each CAS layer has error $\le \epsilon$ and the sorting network has $\ell$ layers, total $L_\infty$ sorting error is bounded by $\epsilon \cdot \ell$. **Crucially, this induction proof holds verbatim for our pure-polynomial CAS gates whenever $|s_\theta(\Delta)| \le 1$ on $[-1, 1]$!**

---

## 5. Pure-Polynomial Sorting in Homomorphic Encryption (CKKS)

In approximate Fully Homomorphic Encryption (specifically CKKS), ciphertexts support **only additions and multiplications** — the exact same computational model as strictly pure Polynomial Neural Networks!

- **Cheon, Kim & Kim (ASIACRYPT 2020; IACR ePrint 2019/1234):** *Efficient Homomorphic Comparison Methods with Optimal Complexity* ([ePrint 2019/1234](https://eprint.iacr.org/2019/1234)). Introduced composite odd polynomials achieving optimal asymptotic multiplicative depth for approximating $\mathrm{sign}(x)$ on $[-1, 1]$:
  $$f_m(x) = \sum_{i=0}^{m} \frac{1}{4^i} \binom{2i}{i} x(1 - x^2)^i.$$
  Explicitly:
  - $f_1(x) = \frac{3}{2}x - \frac{1}{2}x^3$,
  - $f_2(x) = \frac{15}{8}x - \frac{10}{8}x^3 + \frac{3}{8}x^5$,
  - $f_3(x) = \frac{35}{16}x - \frac{35}{16}x^3 + \frac{21}{16}x^5 - \frac{5}{16}x^7$,
  and mixed compositions with faster small-$x$ amplification polynomial $g_3(x) = (4589x - 16577x^3 + 25614x^5 - 12860x^7)/1024$ (verified in Mazzone et al., 2025, Sec. 2.2).
- **Hong, Kim, Choi, Lee & Cheon (IEEE TIFS 2021):** *Efficient Sorting of Homomorphic Encrypted Data with $k$-Way Sorting Network*, IEEE TIFS 16:4389–4404. Combines Cheon et al.'s composite sign polynomials with $k$-way sorting networks, reducing comparison depth from $O(\log_2^2 n)$ to $O(k \log_k^2 n)$.
- **Lee, Lee, No & Kim (IACR ePrint 2020/834; IEEE TDSC 2022):** *Minimax Approximation of Sign Function by Composite Polynomial for Homomorphic Comparison* ([ePrint 2020/834](https://eprint.iacr.org/2020/834); IEEE TDSC 19(6):3711–3727). Replaces fixed binomial $f_m$ with **Remez-optimal minimax composite polynomials** chosen by dynamic programming, minimizing worst-case $L_\infty$ error on $[-1, -\delta] \cup [\delta, 1]$ and cutting depth by ~41% and runtime by ~45%.
- **Mazzone, Everts, Hahn & Peter (USENIX Security 2025; arXiv:2412.15126):** *Efficient Ranking, Order Statistics, and Sorting under CKKS* ([arXiv:2412.15126](https://arxiv.org/abs/2412.15126)). Breaks the $\Omega(\log^2 n)$ swap-network comparison depth barrier by shifting to an **all-to-all $O(1)$-comparison-depth** polynomial architecture:
  1. Simultaneous pairwise comparisons $C_{ij} = \mathrm{cmp}(x_j, x_i) = \frac{1 + s((x_j - x_i)/2)}{2}$.
  2. Fractional rank $r_j = \sum_{i=1}^n C_{ij} + 0.5$.
  3. **Exact Polynomial Tie-Correction Offset (Sec. 4, Alg. 6):** Computes equality matrix $E = 4 C (1 - C)$ (pure quadratic!), upper-triangular sum $U_j = \sum_{i \le j} E_{ij}$, total tie count $T_j = \sum_i E_{ij}$, and tie-resolved rank $r_j^{\text{fixed}} = \sum_i C_{ij} + U_j - 0.5 T_j$.
  4. Parallel polynomial indicator $W_{jk} = \mathrm{ind}_{[k-0.5, k+0.5]}(r_j^{\text{fixed}})$ and extraction $\hat{y}_k = \sum_{j=1}^n W_{jk} x_j$.

---

## 6. Methodological Standards Grounding This Study

To ensure strong predictive and explanatory power without common ML pitfalls, our experimental protocol directly adopts:
- **Lones (2021/2024):** *How to Avoid Machine Learning Pitfalls: A Guide for Academic Researchers* ([arXiv:2108.02497](https://arxiv.org/abs/2108.02497)) — strict split isolation, appropriate baselines, multiple seeds, and transparent reporting of negative results.
- **Lipton & Steinhardt (2018):** *Troubling Trends in Machine Learning Scholarship* ([arXiv:1807.03341](https://arxiv.org/abs/1807.03341)) — separating speculation from mathematical proof/empirical evidence, avoiding conflating architectural tweaks with fundamental gains.
- **Nanda (2022–2024):** *How I Think About My Research Process* ([LessWrong Sequence](https://www.lesswrong.com/s/5GT3yoYM9gRmMEKqL)) — falsifiable hypotheses, discriminating experiments, truth-seeking skepticism, and mechanistic inspection of what networks actually learn.
- **Schulman (2017):** *An Opinionated Guide to ML Research* ([joschu.net](http://joschu.net/blog/opinionated-guide-ml-research.html)) — clean baselines, controlled ablations, and fast diagnostic loops.
- **NeurIPS Paper Checklist Guidelines** ([neurips.cc](https://neurips.cc/public/guides/PaperChecklist)) — error bars, statistical significance, reproducibility, limitations, and compute accounting.

---

## 7. Synthesis: Key Research Gap Identified

Despite deep progress in **three parallel communities**:
1. **Deep PNNs** ($\Pi$-Nets, PolyMLPs, ChebyKANs) have been applied to vision, meshes, and smooth regression — never systematically benchmarked on non-smooth algorithmic list sorting or compared against sorting-structured polynomials.
2. **Differentiable Sorting Networks** (`diffsort`, NeuralSort, SoftSort) rely on *non-polynomial* transcendental relaxations ($\exp$, $\arctan$, division) and focus on ordering-supervision of image encoders rather than pure-polynomial value sorting.
3. **Homomorphic Polynomial Sorting** (Cheon, Hong, Lee, Mazzone) uses *hand-crafted / Remez-fixed* polynomials optimized for **unweighted worst-case $\mathrm{sign}(u)$ error** $|1 - s(u)|$ on $[\delta, 1]$ — never **end-to-end gradient-trained** on data distributions to minimize **value-weighted sorting error** $u|1 - s(u)|$ or allowed to adapt polynomial coefficients per sorting stage!

**Our central scientific opportunity:** Unify these three fields by formulating, auditing, training, and rigorously dissecting a complete ladder of **strictly pure Polynomial Neural Networks** for list sorting — directly testing whether gradient-trained composite PNNs can surpass analytical minimax CKKS polynomials at iso-depth/iso-degree, how architectural inductive bias governs scaling, and where pure polynomials fundamentally succeed or break down.
