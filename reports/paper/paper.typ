#set document(
  title: "Strictly Pure Polynomial Neural Networks for Sorting Continuous Lists",
  author: "Libor \"Burny\" Burian",
  date: datetime(year: 2026, month: 11, day: 25),
)
#set page(
  paper: "a4",
  margin: (x: 1.75cm, y: 1.9cm),
  numbering: "1",
  header: align(right, text(size: 8pt, fill: luma(110))[Strictly Pure Polynomial Neural Networks for Sorting — Libor "Burny" Burian (2026-11-25)]),
)
#set text(font: "New Computer Modern", size: 9.5pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#show link: set text(fill: rgb("#005a9c"))
#show table.cell.where(y: 0): strong
#set table(
  stroke: (x, y) => if y == 0 { (top: 1pt + black, bottom: 0.8pt + black) } else { (bottom: 0.35pt + luma(200)) },
  inset: (x: 4.5pt, y: 3.5pt),
  fill: (_, y) => if y == 0 { rgb("#eef4fa") } else if calc.odd(y) { white } else { rgb("#fafbfc") },
)

#align(center)[
  #text(size: 14.2pt, weight: "bold")[Strictly Pure Polynomial Neural Networks for Sorting Continuous Lists:\ Inductive Bias, Composite Depth Separation, and Fundamental Algebraic Boundaries]
  #v(0.5em)
  #text(size: 11pt, weight: "medium")[Libor "Burny" Burian] \
  #text(size: 9pt, fill: luma(80))[2026-11-25 #h(1em)•#h(1em) #link("https://github.com/BurnyCoder/polynomial-neural-network-sort-list")[GitHub Repository] #h(0.8em)•#h(0.8em) #link("https://huggingface.co/BurnyCoder/polynomial-neural-network-sort-list")[Hugging Face Checkpoints] #h(0.8em)•#h(0.8em) #link("https://burnycoder.github.io/polynomial-neural-network-sort-list/")[Live Browser Pure-Polynomial Sorter]]
]

#v(0.6em)
#rect(fill: rgb("#f4f7fb"), stroke: (left: 2.5pt + rgb("#0072B2")), inset: 9pt, radius: 3pt)[
  #text(weight: "bold")[Abstract.]
  Can a neural network composed _exclusively_ of additions, subtractions, and multiplications—with zero piecewise activations (`ReLU`/`GELU`), zero transcendental functions (`exp`/`atan`/`tanh`/`softmax`), zero divisions, and zero normalization layers—learn to sort lists of continuous real numbers accurately? Prior work is split across three disconnected communities: deep polynomial neural networks ($Pi$-Nets) rarely remove normalization and have not been studied on non-smooth algorithmic sorting; differentiable sorting networks (`diffsort`) rely on non-polynomial sigmoids or Cauchy CDFs; and homomorphic encryption (CKKS) sorting uses fixed analytical minimax polynomials optimizing unweighted worst-case sign error rather than end-to-end value-sorting accuracy.
  We bridge all three domains in a preregistered, mechanically verified (PyTorch FX) empirical study of *178 runs* spanning 17 architectures, 5 random seeds (100 core cells; $N_"test" = 10,000$ lists per run), 10 out-of-distribution (OOD) suites, and 6 causal ablation suites.
  *Positive results:* Replacing discrete compare-and-swap (CAS) gates in Batcher's Bitonic network with end-to-end gradient-trained composite odd Chebyshev polynomials (`PolySortNet`) achieves held-out 5-seed Value MAE `0.00724 ± 0.00000` ($n=8$, 36 params, `95.9%` Exact\@0.05) and `0.00981 ± 0.00008` ($n=16$, 60 params, `81.8%` Exact\@0.05)—beating analytical CKKS composite polynomials (Cheon $f_1^((3))$, HiGHS Remez $L=3$, Cheon/Hong/Mazzone $f_3 compose g_3$) by *1.6×–3.9×*, unstructured pure PNNs ($Pi$-Nets, PolyMLP) by *6.4×–8.3×*, permutation-invariant `DeepSets-Poly` (131k params) by *4.2×–4.6×*, and non-polynomial `ReLU/GELU-MLP` by *4.9×–6.5×* (all Holm-Bonferroni paired Wilcoxon $p_"holm" < 10^(-300)$, 5/5 seeds). A 6-parameter shared-gate `PolySortNet` trained only on $n=8$ extrapolates *16× zero-shot to $n=128$* (`MAE = 0.0200`, Kendall $tau = 0.9978$), beating a 96.6k-parameter `SetTransformer` by *2.65×* on $n=128$, *13,714×* on Shift $+100$ (`0.00726` vs. `99.50`), and on 5-seed Near-Ties $10^(-3)$ (`0.00659` vs. `0.00665`), while preserving exact float64 sum conservation ($2.1 times 10^(-16)$) and shift equivariance ($6.6 times 10^(-14)$). Mechanistic weight inspection reveals that stage-adaptive polynomials autonomously rediscover Bitonic merge-block stride resets.
  *Honest negative results and falsified hypotheses:* Two preregistered hypotheses were explicitly falsified: (1) removing fixed-point anchoring $h(1)=1$ (H6 Part 1) _reduces_ ID MAE by 26–30% via late-stage domain contraction ($s_5(1) = -445$ to $-1038$) at the causal cost of shrinking the upper scale stability radius $R_"stable"$ from `1.3 -> 1.1`; (2) at strict iso-degree $D=9$ (H3 Part 1), a 4-param/stage flat polynomial beats a 2-param/stage composite `L=2, d=3` network on ID MAE before suffering Runge blow-up (`R_stable = 0.0`), whereas iso-parameter composite `L=4, d=3` (deg 81) beats flat deg-9 by *4.28×* and expands $R_"stable"$ to `1.50`. Moreover, non-polynomial `DiffSort` (`0.00041`–`0.00424`) and `SetTransformer` (`0.00218`–`0.00377`) retain sharper in-distribution precision, and every strictly pure polynomial inevitably encounters a finite upper scale blow-up cliff ($s = 1.5$–$2.0$), causally isolated via a non-polynomial `max(|x|)` wrapper (`R_stable: 1.3 -> 5.0`, at the cost of breaking Shift $+100$).
]

= Context, Motivation, and Introduction

Sorting a continuous vector $x = (x_1, dots, x_n) in [-1, 1]^n$ into non-decreasing order $S(x) = (x_((1)), dots, x_((n)))$ is a canonical non-smooth algorithmic map: it is globally 1-Lipschitz, permutation-invariant, shift-equivariant ($S(x + c bold(1)) = S(x) + c bold(1)$), sum-conserving ($sum_k x_((k)) = sum_i x_i$), and non-differentiable along all $binom(n, 2)$ hyperplanes $x_i = x_j$. Because every finite-degree polynomial is $C^infinity$ analytic, no finite polynomial can represent $S(x)$ identically on any open set intersecting $x_i = x_j$. Yet strictly pure polynomials—evaluating _only_ additions, subtractions, and multiplications—are the exact computational substrate of approximate Fully Homomorphic Encryption (CKKS) #link("https://eprint.iacr.org/2019/1234")[(Cheon et al., 2020)], verifiable arithmetic circuits, and polynomial neural architectures #link("https://arxiv.org/abs/2006.13026")[(Chrysos et al., 2021)].

Despite parallel progress in three communities—deep polynomial neural networks ($Pi$-Nets; #link("https://arxiv.org/abs/2006.13026")[Chrysos et al., 2021]), differentiable sorting networks (`diffsort`; #link("https://arxiv.org/abs/2105.04019")[Petersen et al., 2021]\; #link("https://arxiv.org/abs/2203.09630")[Petersen et al., 2022]), and CKKS polynomial sorting (#link("https://eprint.iacr.org/2019/1234")[Cheon et al., 2020]\; #link("https://eprint.iacr.org/2020/834")[Lee et al., 2022]\; #link("https://arxiv.org/abs/2412.15126")[Mazzone et al., 2025])—no prior study has systematically investigated *end-to-end gradient-trained strictly pure Polynomial Neural Networks (PNNs)* for continuous list sorting across the full architectural inductive-bias spectrum, nor directly benchmarked learned composite PNNs against analytical CKKS minimax polynomials at matched polynomial degree and multiplicative depth.

== Explicit Statement of Novelty

1. *Unification & Mechanical Purity Enforcement:* We formulate and mechanically audit (via PyTorch FX graph tracing, zero hidden activations/divisions/norms) a 5-tier ladder of strictly pure PNN sorters: `PolyMLP`, $Pi$-Net (`NCP-Skip`), `DeepSets-Poly`, constant-depth `PolyPairwiseRank` (with Mazzone's exact quadratic tie correction $E = 4C(1-C)$), and our end-to-end trained composite polynomial sorting network `PolySortNet`.
2. *Gradient-Trained Composite Polynomials Surpass Analytical CKKS at Iso-Degree:* Analytical CKKS polynomials minimize unweighted worst-case sign error $sup_(u in [delta, 1]) |1 - s(u)|$, whereas value sorting minimizes *value-weighted* comparator error $e_"val"(Delta) = |Delta| dot |1 - s(|Delta|)|$ across non-uniform, stage-shrinking pairwise gap distributions. Training composite odd Chebyshev coefficients end-to-end achieves 1.6×–3.9× lower Value MAE than Cheon $f_1^((3))$, HiGHS Remez $L=3$, and Cheon/Hong/Mazzone $f_3 compose g_3$ using only 3–60 learnable scalars.
3. *Mechanistic Discoveries & Honest Falsifications:* Direct float64 transfer inspection uncovers three structural phenomena: (a) stage-adaptive polynomials autonomously rediscover Bitonic merge-block stride resets; (b) removing $h(1)=1$ fixed-point anchoring improves ID MAE by 26–30% via late-stage domain contraction ($s_5(1) = -445..-1038$) while sacrificing scale stability ($R_"stable": 1.3 -> 1.1$); and (c) deep low-degree composition (`L=4, d=3`, deg 81) dominates iso-parameter flat degree-9 by 4.28× on MAE while simultaneously avoiding Runge boundary explosion (`R_stable: 0.0 -> 1.50`).

= Background and Mathematical Foundations

== Approximation of $|u|$: Flat vs. Composite Polynomials
For two scalars $a, b in [-1, 1]$, define midpoint $mu = (a+b)/2$ and half-difference $Delta = (b-a)/2 in [-1, 1]$. Then
$ min(a, b) = mu - |Delta|, quad max(a, b) = mu + |Delta|. $
By Bernstein's theorem (1914) and high-precision Remez computations by Varga & Carpenter (1985, 1987), the best uniform degree-$D$ polynomial approximation to $|u|$ on $[-1, 1]$ decays only algebraically: $E_D(|u|) tilde.op beta / D$ with $beta approx 0.2801694990$. Consequently, flat polynomials require $D = Theta(1/epsilon)$ parameters and degree. Conversely, composing $L$ low-degree odd polynomials $s(u) = (h_L compose dots.c compose h_1)(u)$ of degree $d = 2m+1$ attains total degree $D = (2m+1)^L$ with only $O(L m)$ parameters and sequential multiplications (#link("https://eprint.iacr.org/2019/1234")[Cheon et al., 2020]\; #link("https://eprint.iacr.org/2020/834")[Lee et al., 2022]).

== Algebraic Invariants & Rolle's Theorem Trade-Off
Replacing $|Delta|$ with $Delta dot s(Delta)$ for any odd polynomial $s(-u) = -s(u)$ defines a *pure-polynomial compare-and-swap (CAS) gate*:
$ hat(min)_s (a, b) = mu - Delta dot s(Delta), quad hat(max)_s (a, b) = mu + Delta dot s(Delta). $
This gate satisfies four exact algebraic invariants for *every* odd polynomial $s$:
1. *Exact Sum Conservation:* $hat(min)_s (a, b) + hat(max)_s (a, b) = 2 mu = a + b$, so by induction across any sorting network $sum_k hat(y)_k = sum_i x_i$.
2. *Exact Shift Equivariance:* Because $Delta = ((b+c)-(a+c))/2 = (b-a)/2$ is strictly invariant to additive shifts $c in RR$, $hat(min)_s (a+c, b+c) = hat(min)_s (a,b) + c$, extending by induction to $f_theta(x + c bold(1)) = f_theta(x) + c bold(1)$ for arbitrary $c$ (e.g., $c = 100$).
3. *Domain-Boundedness:* Whenever $|s(u)| <= 1$ on $[-1, 1]$, both outputs lie strictly inside $[min(a,b), max(a,b)] subset.eq [-1, 1]$, keeping wire activations bounded across arbitrary sorting depth without normalization and inheriting Theorem 9 of #link("https://arxiv.org/abs/2203.09630")[Petersen et al. (2022)]: if single-gate error $Delta|1 - s(Delta)| <= epsilon$, an $ell$-stage network has $L_infinity$ error $<= epsilon dot ell$.
4. *Rolle's Theorem Monotonicity Constraint:* By Theorem 5 of #link("https://arxiv.org/abs/2203.09630")[Petersen et al. (2022)], strict global monotonicity requires $f'(x) in Omega(1/x^2)$ (satisfied by Cauchy CDF $arctan$, impossible for bounded-degree polynomials). Indeed, if $s(0)=0$, $s(1)=1$, and $s(u) < 1$ on $(0,1)$, then $hat(min)_s (2, 0) = 0$ while $hat(min)_s (x, 0) > 0$ on $(0, 2)$, so by Rolle's theorem $partial_x hat(min)_s (x, 0) < 0$ somewhere on $(0, 2)$. Exact boundary fidelity at $u=1$ thus precludes strict global monotonicity, explaining why pure polynomial sorters trade a small fraction of near-tie inversions against sharp value fidelity.
5. *Unavoidable Upper Scale Blow-Up:* Any non-affine polynomial $p$ of degree $D >= 2$ diverges as $O(||x||^D)$ outside its contractive basin, guaranteeing a finite upper scale stability radius $R_"stable" < infinity$—unlike transcendental `diffsort` ($exp$/$arctan$), which saturates globally as $|Delta| -> infinity$.

= Method

== `PolySortNet` and Fixed-Point Odd Chebyshev Parameterization
`PolySortNet` instantiates a fixed Knuth 0-1 verified sorting network (default: Batcher's Bitonic network, $ell = frac(1,2) log_2 n (log_2 n + 1)$ parallel stages, $n/2$ gates/stage) whose CAS gates use an $L$-layer composite polynomial $s_k (Delta) = (h_(k,L) compose dots.c compose h_(k,1))(Delta)$. Each layer uses our *fixed-point odd Chebyshev parameterization* with $m$ learnable scalars $c_(k,ell,0), dots, c_(k,ell,m-1)$:
$ h_(k,ell)(u) = u + u(1 - u^2) sum_(j=0)^(m-1) c_(k,ell,j) T_(2j)(u), $
where even Chebyshev terms are computed strictly via additions and multiplications using $T_(2j)(u) = T_j (2u^2 - 1)$. This form algebraically enforces $h(0)=0$, $h(plus.minus 1) = plus.minus 1$, odd symmetry, and exact subsumption of both the identity map ($c=0$) and Cheon et al.'s (2020) analytical $f_m (u) = sum_(i=0)^m 4^(-i) binom(2i, i) u(1-u^2)^i$ (verified symbolically in SymPy to $< 10^(-13)$). We evaluate two weight-sharing regimes:
- *`PolySortNet-Shared`:* One shared comparator across all stages ($L dot m$ total parameters: *3* for $m=1, L=3$, deg 27; *6* for $m=2, L=3$, deg 125). Supports *zero-shot length generalization* to any power-of-2 length $n in {4, 8, 16, 32, 64, 128}$.
- *`PolySortNet-Adaptive`:* Distinct coefficients per parallel stage ($ell dot L dot m$ total parameters: 18/30 for $m=1$, 36/60 for $m=2$ at $n in {8, 16}$), allowing each stage to specialize to its local gap distribution.

#figure(
  image("figures/fig1_architecture.png", width: 100%),
  caption: [*PolySortNet architecture and 5-tier Pure-PNN inductive-bias ladder.* (A) Fixed Bitonic wiring guarantees zero-shot length extensibility, sum conservation, and shift equivariance. (B) Strictly pure-polynomial CAS gate using $L$-layer composite fixed-point odd Chebyshev blocks (mechanically FX-verified). (C) Five pure-PNN tiers evaluated from unstructured `PolyMLP` up to `PolySortNet`.],
)

== Full Architecture Ladder, Analytical CKKS, Controls, and Baselines
Every model tagged `is_pure_polynomial=True` is mechanically audited via `torch.fx.symbolic_trace` (`src/pnn_sort/audit.py`), whitelisting only `+`, `-`, `*`, constant scalar division, and linear/index permutations while rejecting all non-polynomial ops (verified by 62/62 passing unit tests including 18 purity proofs and 5 negative rejection tests):
1. *Unstructured & Symmetric Pure PNNs:* `PolyMLP` ($x^2+x$ activations, 37–39k params); `PiNet-NCP-Skip` (#link("https://arxiv.org/abs/2006.13026")[Chrysos et al., 2021], Eq. 10, strictly without BN/`tanh`, 202–206k params); `DeepSets-Poly` (#link("https://arxiv.org/abs/1703.06114")[Zaheer et al., 2017], Chebyshev power-sum encoder with latent width $>= 2n$ respecting #link("https://arxiv.org/abs/1901.09006")[Wagstaff et al., 2019], 131–132k params); `PolySortNet-LearnedWire` (orthogonal linear routers + poly CAS, 466–2.8k params); and `PolyPairwiseRank` (#link("https://arxiv.org/abs/2412.15126")[Mazzone et al., 2025] constant-depth all-to-all ranking with exact quadratic tie correction $E=4C(1-C)$, 12 params).
2. *Zero-Training Analytical CKKS Baselines (Pure Poly, 0 trained params):* Cheon $f_1^((3))$ (deg 27), Cheon $f_2^((2))$ (deg 25), Cheon/Hong/Mazzone mixed $f_3 compose g_3$ (deg 49, exact coefficients from #link("https://arxiv.org/abs/2412.15126")[Mazzone et al., 2025] Sec. 2.2), and #link("https://eprint.iacr.org/2020/834")[Lee et al. (2022)] Algorithm 2 composite Remez minimax solved via exact HiGHS convex LP (`scipy.optimize.linprog`) at $L=3, m=1$ ($delta=0.10$) and $L=3, m=2$ ($delta=0.03$).
3. *Controls:* `Identity`, OLS `LinearControl`, untrained random `PolySortNet`/`PiNet`, `ShuffledTarget` leakage control (permuted targets), and `RandomFixedWiring` topology control.
4. *Non-Polynomial Reference Baselines:* `DiffSort-Logistic` ($sigma(beta Delta)$) and monotonic `DiffSort-Cauchy` ($frac(1, pi) arctan(beta Delta) + frac(1,2)$) (#link("https://arxiv.org/abs/2203.09630")[Petersen et al., 2022]), `ReLU/GELU-MLP` (35–37k params), `SetTransformer` (#link("https://arxiv.org/abs/1810.00825")[Lee et al., 2019], 96.6k params), and non-poly `ScaleNorm` wrapper (Ablation A6).

= Experimental Setup and Statistical Rigor

All hypotheses (`H1–H6`), metrics, splits, and falsification criteria were locked in `reports/preregistration.md` and pushed to GitHub (`879258c`) _prior to executing any training or evaluation_.
- *Zero-Leakage Datasets:* Independent SHA-256 keyed streams generate $200,000$ training lists, $10,000$ validation lists, and $10,000$ test lists i.i.d. from $"Uniform"[-1, 1]^n$, plus 10 OOD stress suites ($2,000$ lists each): Gaussian, Bimodal, Skewed Beta, Discrete Integers, 50% Duplicates, Near-Ties ($delta in {10^(-2), 10^(-3), 10^(-4)}$), Edge Sorted/Reversed, Shift ($c in {1, 10, 100}$), Scale ($s in [0.1, 5.0]$), and Zero-Shot Lengths ($n in {4..128}$).
- *Training Protocol & Compute:* AdamW with cosine decay, warmup, and gradient clipping on a single NVIDIA RTX 5070 Laptop GPU (8 GB VRAM, PyTorch 2.11+cu128). Compact sorting networks train with `batch_size=4096, lr=8e-3, epochs=50` (logged WDDM throughput optimization, verified against `bs=512` and `float64` in Stage 4E); dense networks train with `batch_size=2048, lr=2e-3, epochs=60`. All 178 runs completed with atomic resumable checkpoints ($approx 5$ hours total GPU wall-clock).
- *Metrics & Statistical Testing:* We co-report Value MAE, RMSE, Wasserstein-1 $W_1$, Sum Error, Monotonicity Rate, Kendall's $tau$, Value-Matched Exact Sort Rate (`Ex@0.05`: monotonic AND $L_infinity < 0.05$, preventing constant-output monotonicity gaming), and Upper Scale Stability Radius $R_"stable"$ (largest $s >= 1.0$ with $"MAE"/s < 0.05$). All 10 core models are replicated across *5 independent seeds* (`42, 123, 456, 789, 1024`; 100 cells) and evaluated with 10,000-resample bootstrap 95% CIs, two-sided paired Wilcoxon signed-rank tests (50,000 pooled paired lists per 5-seed test), rank-biserial effect sizes $r_"rb"$, and Holm-Bonferroni correction across all 34 planned tests.

= Empirical Results (Positive and Negative Findings)

== Core 5-Seed Benchmarks ($n = 8$ and $n = 16$)

Table 1 reports the complete 5-seed benchmarks alongside single-seed analytical CKKS and control references (visualized in Figure 2). Key findings:
1. *Inductive Bias Ladder (H1, 5/5 seeds, $p_"holm" < 10^(-300)$):* Pure-polynomial `PolySortNet-Adaptive (m=2)` (36/60 params) achieves 5-seed MAE `0.00724 ± 0.00000` ($n=8$, `95.9%` Ex\@0.05) and `0.00981 ± 0.00008` ($n=16$, `81.8%` Ex\@0.05)—outperforming `DeepSets-Poly` (131k params) by *4.56× / 4.20×*, `PiNet-NCP-Skip` (202–206k params) by *7.49× / 6.35×*, `PolyMLP` by *8.26× / 6.61×*, and non-poly `ReLU/GELU-MLP` by *4.90× / 6.48×*. At $n=16$, all unstructured networks (`PiNet`, `PolyMLP`, `ReLU-MLP`, `PolyPairwiseRank`) collapse to `0.0%` Ex\@0.05 despite high vacuous monotonicity (`98–99%`), confirming that monotonicity alone is severely misleading without value-matched error!
2. *Honest H1 Cubic Caveat:* At lower comparator degree $m=1$ (deg 27), 18-param `PolySortNet-Adaptive` slightly trails 131k-param `DeepSets-Poly` on $n=8$ ID MAE (`0.03327` vs. `0.03300`, `DeepSets-Poly` wins 4/5 seeds, $r_"rb" = +0.056$) before overtaking it 5/5 seeds at $n=16$ (`0.03693` vs. `0.04122`) and dominating on Duplicates (`0.0324` vs. `0.0746`) and Shift $+100$ (`0.0334` vs. `inf`).
3. *Four Orders of Magnitude Lower Seed Variance:* `PolySortNet-Adaptive (m=2)` has seed SD `5.76e-7` at $n=8$ (`CV = 0.008%`)—*9,078× smaller* than `PiNet-NCP-Skip` (`5.23e-3`) and *374× smaller* than `SetTransformer` (`2.16e-4`).
4. *Surpassing Analytical CKKS at Iso-Depth (H2):* At exact iso-degree ($L=3, m=1$, deg 27), learned `PolySortNet-Adaptive` cuts MAE by *3.42× / 3.94×* vs. Cheon $f_1^((3))$ (`0.1136 / 0.1457`) and *1.60× / 2.00×* vs. HiGHS Remez $L=3$ (`0.0531 / 0.0740`). At $m=2$ (deg 125), `PolySortNet-Adaptive (m=2)` beats Cheon/Hong/Mazzone mixed $f_3 compose g_3$ (`0.0279 / 0.0375`) by *3.85×* at both lengths, while unscaled analytical Remez $m=2$ diverges (`inf`). Stage-adaptive gains over shared gates grow monotonically with list length: `-6.2%` ($n=4$) $->$ `-11.9%` ($n=8$) $->$ `-23.7%` ($n=16$) $->$ `-31.5%` ($n=32$).
5. *Honest Non-Poly Ceiling vs. OOD Reversal:* Non-polynomial `DiffSort-Logistic` (`0.00041 / 0.00069`), `DiffSort-Cauchy` (`0.00306 / 0.00424`), and `SetTransformer` (`0.00218 / 0.00377`) remain more accurate in-distribution. *Yet* averaged across all 5 seeds at $n=8$, 36-parameter pure-polynomial `PolySortNet-Adaptive (m=2)` *outperforms* 96.6k-parameter `SetTransformer` on *Near-Ties $10^(-3)$* (`0.00659` vs. `0.00665`), *50% Duplicates* (`0.00660` vs. `0.00737`), and *Shift $+100$* (`0.00726` vs. `99.50`, *13,714×*)!
6. *Constant-Depth Crossover (H4) & Leakage Verification:* `PolyPairwiseRank` (12 params) beats unstructured `PiNet`/`PolyMLP` at $n=8$ (`0.04232` vs. `0.05426`/`0.05984`) before degrading sharply at $n=16$ (`0.07380`, `0.11%` Mono) and losing shift equivariance (`Shift +100 = 9.93–12.10` because indicator weights do not sum identically to 1). `ShuffledTarget` scores `0.210 / 0.282` (`0.0%` Ex\@0.05), confirming zero data leakage.

#figure(
  text(size: 7.9pt)[#table(
    columns: (2.35fr, 0.55fr, 0.9fr, 1.45fr, 0.85fr, 1.45fr, 0.85fr, 0.95fr, 0.65fr),
    align: (left, center, right, right, right, right, right, right, right),
    [Model], [Poly], [Params], [MAE ($n=8$)], [Ex05 ($8$)], [MAE ($n=16$)], [Ex05 ($16$)], [Shift $+100$], [$R_"stab"$],
    [*PolySortNet-Adapt ($m=2$)* (5s)], [*Yes*], [36/60], [*0.00724 ± .00000*], [*95.9%*], [*0.00981 ± .00008*], [*81.8%*], [*0.0073*], [1.00],
    [*PolySortNet-Shared ($m=2$)* (1s)], [*Yes*], [6/6], [0.00794], [92.2%], [0.01108], [72.5%], [*0.0079*], [1.00],
    [*PolySortNet-Adapt $L=4$* (1s)], [*Yes*], [24/40], [0.01322], [91.8%], [0.01631], [61.4%], [*0.0133*], [*1.50*],
    [Analytical Cheon $f_3 compose g_3$], [Yes], [0], [0.02788], [27.5%], [0.03749], [0.3%], [0.0279], [1.10],
    [DeepSets-Poly (5s)], [Yes], [131k], [0.03300 ± .00069], [21.6%], [0.04122 ± .00014], [0.7%], [`inf`], [1.05],
    [*PolySortNet-Adapt ($m=1$)* (5s)], [*Yes*], [18/30], [0.03327 ± .00002], [16.3%], [0.03693 ± .00001], [1.0%], [*0.0334*], [1.30],
    [*PolySortNet-Shared ($m=1$)* (5s)], [*Yes*], [3/3], [0.03776 ± .00000], [9.8%], [0.04842 ± .00000], [0.0%], [*0.0377*], [1.30],
    [PolyPairwiseRank (5s)], [Yes], [12/12], [0.04232 ± .00038], [13.6%], [0.07380 ± .00012], [0.0%], [9.934], [1.01],
    [Analytical Remez $L=3$], [Yes], [0], [0.05309], [1.7%], [0.07400], [0.0%], [0.0533], [0.00],
    [PiNet-NCP-Skip (5s)], [Yes], [202k], [0.05426 ± .00523], [2.7%], [0.06236 ± .00003], [0.0%], [2.8e17], [0.21],
    [PolyMLP ($x^2+x$) (5s)], [Yes], [37k], [0.05984 ± .00293], [1.5%], [0.06482 ± .00178], [0.0%], [`inf`], [0.00],
    [PolySortNet-LearnedWire], [Yes], [466/2.8k], [0.09666], [0.0%], [0.07535], [0.0%], [`inf`], [0.00],
    [Analytical Cheon $f_1^((3))$], [Yes], [0], [0.11362], [0.0%], [0.14568], [0.0%], [0.1131], [0.00],
    [Linear / Shuffled Ctrl], [Yes], [72/18], [0.1346 / 0.2101], [0.0%], [0.0986 / 0.2817], [0.0%], [26.6 / 0.21], [0.00],
    [DiffSort-Logistic (non-poly)], [No], [1], [0.00041], [99.9%], [0.00069], [99.2%], [0.0004], [5.00],
    [SetTransformer (non-poly, 5s)], [No], [96.6k], [0.00218 ± .00022], [99.9%], [0.00377 ± .00045], [99.7%], [99.502], [1.50],
    [DiffSort-Cauchy (non-poly, 5s)], [No], [1], [0.00306 ± .00000], [100%], [0.00424 ± .00000], [99.9%], [0.0031], [5.00],
    [ReLU/GELU-MLP (non-poly, 5s)], [No], [35k], [0.03549 ± .00122], [15.2%], [0.06361 ± .00054], [0.0%], [27.482], [1.42],
  )],
  caption: [*Primary benchmark table across strictly pure PNNs, analytical CKKS polynomials, controls, and non-polynomial baselines* (`5s` = 5-seed mean ± SD over $5 times 10,000$ lists; `Ex05` = monotonic AND $L_infinity < 0.05$; `Shift +100` shown for $n=8$). Note that tuning `PiNet` LR to `0.01` improves its $n=8$ MAE to `0.04232` (Appendix B), still 5.85× worse than `PolySortNet-Adapt (m=2)`.],
)

#figure(
  image("figures/fig2_main_comparison.png", width: 100%),
  caption: [*Held-out Value MAE (log scale) across all architectures at $n=8$ and $n=16$* with 5-seed SD error bars and `Ex@0.05` annotations. Strictly pure `PolySortNet` (blue) fills the gap between unstructured PNNs/CKKS polynomials and transcendental upper bounds.],
)

== Mechanistic Inspection: Value-Weighted Error & Rediscovery of Bitonic Stride Resets

Evaluating trained comparator polynomials directly in `float64` (Figure 3) explains *mechanistically* why end-to-end training beats analytical CKKS polynomials by up to 3.9×:
1. *Overshoot-and-Contract Rediscovery:* `PolySortNet-Shared (m=1)` learns monotonically decreasing layer coefficients `[[1.4017], [0.8521], [0.7530]]`, yielding origin slope $s'(0) = 7.80$ vs. Cheon $f_1^((3))$'s $1.5^3 = 3.38$. Early layers amplify small pairwise gaps aggressively (controlled intermediate overshoot analogous to Cheon's hand-crafted $g_3$), while the final layer contracts values back to $plus.minus 1$, slashing single-gate peak value error $max_Delta Delta|1 - s(Delta)|$ from `0.080` (Cheon $f_1^((3))$) and `0.084` (Remez $L=3$) down to `0.034` ($m=1$) and `0.011` ($m=2$).
2. *Autonomous Rediscovery of Bitonic Merge-Block Strides (Figure 3C):* Rather than rising monotonically across all stages, stage-adaptive origin slope $s'(0)$ exhibits sharp *resets* at Stage 3 ($n=8$) and Stages 1, 3, and 6 ($n=16$)—the _exact_ entry stages of each new Bitonic merge block where comparator stride doubles and incoming pairwise gaps $|Delta|$ abruptly jump back up—before climbing monotonically as stride halves ($8 -> 4 -> 2 -> 1$).

#figure(
  image("figures/fig3_mechanistic_comparators.png", width: 100%),
  caption: [*Mechanistic comparator analysis.* (A) Transfer curve $s(Delta)$. (B) Single-gate value error $Delta|1 - s(Delta)|$ showing why trained PNNs beat analytical uniform-sign polynomials. (C) Stage-adaptive origin slope $s'(0)$ autonomously dipping at every Bitonic long-stride merge boundary.],
)

== Zero-Shot Length Extrapolation, Output Examples, and the Scale Blow-Up Cliff (H5)

Figure 4 displays concrete output examples on both *positive scenarios* (in-distribution Uniform and Shift $+100$) and *honest negative failure modes* (Scale $2.0 times$ blow-up and Random-Wiring collapse). Figure 5 quantifies both OOD boundaries:
- *Zero-Shot $16 times$ Length Scaling (Positive):* Trained solely on $n=8$, 6-parameter `PolySortNet-Shared (m=2)` extrapolates smoothly to $n=128$ (28 Bitonic stages) with `MAE = 0.02001`, $tau = 0.9978$—beating 96.6k-param `SetTransformer` (`0.05311`, *2.65×*) and analytical Cheon $f_3 compose g_3$ (`0.05135`, *2.57×*) while trailing only transcendental `DiffSort-Cauchy` (`0.00858`). In `float64`, sum conservation error is $2.1 times 10^(-16)$ and Shift $+100$ equivariance $L_infinity$ is $6.6 times 10^(-14)$.
- *Scale-Extrapolation Cliff (Predicted Negative Boundary):* Every strictly pure polynomial exhibits an abrupt divergence cliff outside $[-1, 1]$: `m=2` blows up at $s = 1.5$ ($R_"stable" = 1.00$), `m=1` at $s = 2.0$ ($R_"stable" = 1.30$), and `L=4, d=3` at $s = 2.0$ ($R_"stable" = 1.50$). Wrapping the exact same polynomial in non-polynomial `x / max(|x|)` (Ablation A6) immediately restores $R_"stable" = 5.00$ (`Scale 5x` MAE `0.146 / 0.176`), proving H5 causally—while exposing an inverse symmetry trade-off: uncentered `max(|x|)` scaling breaks Shift $+100$ equivariance (`0.033 -> 0.455`) by shrinking shifted differences $101 times$.

#figure(
  image("figures/fig4_output_examples.png", width: 100%),
  caption: [*Concrete output examples across positive wins and negative failures ($n=8$).* Row 1 (ID): 36-param `PolySortNet-Adapt m=2` closely matches ground truth (`0.0048`). Row 2 (Shift $+100$): `PolySortNet` retains exact shift equivariance (`0.0048`) while `PiNet` diverges ($1.9 times 10^(17)$) and `SetTransformer` collapses to `mean ~0.8` (`MAE = 99`). Row 3 (Scale $2 times$): pure polynomial blows up (`INF`), whereas non-poly `DiffSort-Cauchy` stays exact (`0.0031`).],
)

#figure(
  image("figures/fig5_zeroshot_and_scale_cliff.png", width: 100%),
  caption: [*OOD generalization boundaries (H5).* (A) Zero-shot list-length extrapolation from $n=4$ to $n=128$ (16× training length). (B) Scale extrapolation sweep $s in [0.1, 5.0]$ isolating the sharp pure-polynomial divergence cliff vs. A6 non-poly scaling and `DiffSort-Cauchy`.],
)

== Causal Ablations (A1–A6) and Honest Hypothesis Falsifications

Table 2 and Figure 6 summarize all 30 Stage-4D causal interventions against `PolySortNet-Adaptive` ($L=3, m=1$, fixed-point odd Chebyshev, Bitonic wiring, Cheon init, MSE):
1. *A1 Parameterization — H6 Part 1 FALSIFIED on ID MAE, Refined as Pareto Frontier:* Removing $h_k (1) = 1$ (`unconstrained_odd_cheby`, `raw_monomial`, `general_cheby`) *reduces* ID MAE by 26–30% (`0.0234 / 0.0273` vs. `0.0333 / 0.0369`, $r_"rb" = +0.90$ and $+0.94$) and cuts $n=16$ Bimodal MAE 2.2× (`0.0389` vs. `0.0858`). *Mechanism:* Because late Bitonic stages only receive small $|Delta| lt.tilde 0.4$, unconstrained polynomials contract their valid domain to boost Stage-5 slope ($9.58 -> 14.3$), letting $s_5 (1.0)$ explode to $-444.7$ through $-1037.7$ and causing immediate blow-up at Scale $1.3 times$ ($R_"stable": 1.3 -> 1.1$). Fixed-point anchoring is thus a *scale-stability regularizer*, not an ID-MAE maximizer.
2. *A2 Depth Separation — H3 Part 1 FALSIFIED at Iso-Degree $D=9$, CONFIRMED at Iso-Parameter:* At strict iso-degree $D=9$, 4-param/stage flat `L=1, d=9` (`0.0566 / 0.0674`) beats 2-param/stage composite `L=2, d=3` (`0.0771 / 0.0831`) because the 2D composite manifold is strictly contained in the 4D flat degree-9 space—yet flat deg-9 suffers severe Runge boundary explosion ($s(1.1)=1.92 => R_"stable" = 0.00$). At *equal parameter budget (4 params/stage)*, composite *`L=4, d=3` (degree 81)* crushes flat deg-9 by *4.28× / 4.13×* (`0.0132 / 0.0163`, `Ex@0.05 = 91.8% / 61.4%` vs. `1.4% / 0.0%`) while simultaneously expanding stability to *$R_"stable" = 1.50$* (`|s(1.5)| <= 1.005`), strictly dominating quintic `m=2` (`R_stable = 1.00`) on stability!
3. *A3 Init, A4 Wiring (H6 Part 2), and A5 Loss:* Cheon init beats identity ($c=0$) and random init by 4–8% MAE with 0% divergence across all three. Replacing valid wiring with `RandomFixed` causes total collapse (`0.00%` Mono, MAE `0.386 / 0.436`), proving wiring causality. *Batcher Odd-Even Merge* beats Bitonic by 10–13% MAE (`0.0288 / 0.0333`) and halves Scale $1.5 times$ error (`0.0839` vs. `0.1532`). *Habermann Odd-Even Transposition* hits `99.30% / 96.77%` Mono ($tau = 0.9988$ and $0.9983$) at the cost of a 2.74× reversed-input asymmetry at $n=16$ (`EdgeR = 0.1349` vs. `EdgeS = 0.0493`). Adding a monotonicity hinge (`w=0.1`, A5) lifts $n=16$ Mono by +6.71 pp (`75.81% -> 82.52%`) at +3.7% MAE cost.

#figure(
  text(size: 8.1pt)[#table(
    columns: (2.8fr, 0.55fr, 0.85fr, 1.0fr, 1.0fr, 0.95fr, 0.95fr, 0.65fr),
    align: (left, center, right, right, right, right, right, right),
    [Ablation Intervention (vs. Ref $L=3, m=1$)], [Poly], [Params], [MAE ($8$)], [MAE ($16$)], [Mono ($16$)], [Ex05 ($16$)], [$R_"stab"$],
    [*Reference:* `PolySortNet-Adapt` (Fixed-Pt, Bitonic, MSE)], [Yes], [18 / 30], [0.03325], [0.03694], [75.81%], [0.95%], [1.30],
    [A1: Unconstrained Odd Cheby (*H6.1 falsified*)], [Yes], [36 / 60], [*0.02342*], [0.02783], [84.66%], [9.21%], [1.10],
    [A1: Raw Monomial Basis (*H6.1 falsified*)], [Yes], [36 / 60], [0.02381], [*0.02731*], [88.16%], [11.23%], [1.10],
    [A1: General Cheby (no odd symmetry)], [Yes], [72 / 120], [0.02341], [0.02809], [82.36%], [8.32%], [1.10],
    [A2: Flat `L=1, d=9` (4p/st, deg 9; *Runge cliff*)], [Yes], [24 / 40], [0.05662], [0.06742], [90.41%], [0.00%], [0.00],
    [A2: Comp `L=2, d=3` (2p/st, deg 9; *H3.1 falsified*)], [Yes], [12 / 20], [0.07712], [0.08312], [90.26%], [0.00%], [0.00],
    [A2: *Comp `L=4, d=3` (4p/st, deg 81; Best Frontier)*], [Yes], [24 / 40], [*0.01322*], [*0.01631*], [81.47%], [*61.38%*], [*1.50*],
    [A3: Identity Init ($c=0$) / Random Init], [Yes], [18 / 30], [0.0348 / 0.0344], [0.0389 / 0.0399], [73.3% / 67.3%], [0.6% / 0.5%], [1.30],
    [A4: *Batcher Odd-Even Merge Wiring*], [Yes], [18 / 30], [*0.02884*], [*0.03326*], [80.85%], [2.29%], [1.30],
    [A4: Habermann Odd-Even Transposition], [Yes], [24 / 48], [0.03830], [0.04114], [*96.77%*], [0.41%], [1.30],
    [A4: Random Fixed Wiring (*Negative Control*)], [Yes], [18 / 30], [0.38611], [0.43629], [0.00%], [0.00%], [0.00],
    [A5: Huber Loss / MSE + Mono Hinge ($lambda = 0.1$)], [Yes], [18 / 30], [0.0322 / 0.0333], [0.0364 / 0.0383], [73.1% / 82.5%], [1.4% / 0.7%], [1.30],
    [A6: Non-Poly `max(|x|)` Wrapper (*H5 Causal Proof*)], [No], [18 / 30], [0.02934], [0.03512], [76.34%], [1.56%], [*5.00*],
  )],
  caption: [*Causal ablation matrix (Stage 4D, 30 runs, seed 42).* Every row modifies a single causal factor relative to `PolySortNet-Adapt`.],
)

#figure(
  image("figures/fig6_ablations_and_training.png", width: 100%),
  caption: [*Causal ablations (A1–A4) and validation convergence.* (A) Iso-degree vs. iso-parameter depth separation. (B) Accuracy-vs-$R_"stable"$ Pareto frontier. (C) Wiring topology causality. (D) Validation MAE trajectories on $n=16$.],
)

= Claims, Evidence Strength Taxonomy, and Preregistration Verdict

Following truth-seeking scientific reporting standards (#link("https://arxiv.org/abs/1807.03341")[Lipton & Steinhardt, 2018]\; #link("https://www.lesswrong.com/s/5GT3yoYM9gRmMEKqL")[Nanda, 2022]), Table 3 classifies every claim by confidence tier and exact evidence modality:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, left),
    [Scientific Claim / Hypothesis], [Confidence Tier], [Evidence Type], [Grounding & Honest Qualification],
    [Exact sum conservation & shift equivariance of odd-CAS `PolySortNet`], [*Guarantee*], [Algebraic Proof + f64 Audit], [Holds for all $x in RR^n, c in RR$; verified to $2.1 times 10^(-16)$ and $6.6 times 10^(-14)$ in float64.],
    [Finite scale stability radius for all non-affine pure PNNs (H5)], [*Guarantee* + Causal], [Math Proof + A6 Intervention], [Polynomial growth $O(||x||^D)$ forces divergence; A6 `max(|x|)` restores $R_"stable" = 5.0$.],
    [H1: `PolySortNet-m2` beats unstructured PNNs & `DeepSets-Poly`], [*Systematic* (narrow to $m >= 2$ or $n >= 16$)], [5-Seed Paired Wilcoxon], [5/5 seeds, $4.2 times - 8.3 times$ MAE reduction, $r_"rb" = -1.000, p_"holm" < 10^(-300)$. *Caveat:* cubic $m=1$ ties `DeepSets-Poly` at $n=8$.],
    [H2: Learned composite PNNs beat analytical CKKS at iso-degree], [*Systematic*], [Iso-Depth Benchmark + Weight Audit], [1.6×–3.9× lower MAE vs. Cheon $f_1^((3))$, Remez $L=3$, and $f_3 compose g_3$ ($p_"holm" < 10^(-300)$).],
    [H3: Composite depth separation vs. flat polynomial], [*Refined* (H3.1 *Falsified*; Iso-Param Confirmed)], [Causal Ablation A2], [Flat `L=1, d=9` beats `L=2, d=3` at iso-degree $D=9$; iso-param `L=4, d=3` beats flat deg-9 by 4.28× and widens $R_"stable"$ $0 -> 1.50$.],
    [H4: `PolyPairwiseRank` crossover ($n=8$ vs. $n=16$)], [*Systematic*], [5-Seed Paired Wilcoxon], [Beats `PiNet`/`PolyMLP` at $n=8$; degrades to `0.0738` (`0.11%` Mono) at $n=16$ ($7.52 times$ behind `PolySortNet-m2`).],
    [H6.1: Fixed-point anchor improves ID MAE], [*FALSIFIED*], [Causal Ablation A1 + Transfer Audit], [Unconstrained odd polynomials cut MAE 26–30% via late-stage domain contraction ($s_5(1)=-445..-1038$), trading $R_"stable": 1.3 -> 1.1$.],
    [H6.2: Sorting-network wiring is causally necessary], [*Causal Proof*], [Ablation A4], [Random fixed wiring fails (`0.00%` Mono, 11.8× worse MAE); Batcher Odd-Even Merge beats Bitonic 10–13%.],
  ),
  caption: [*Calibrated claims and evidence taxonomy* distinguishing mathematical guarantees, causal interventions, systematic multi-seed results, and explicit preregistration falsifications.],
)

= Related Work, Discussion, and Limitations

*Related Work.* Classical polynomial architectures include GMDH (Ivakhnenko, 1971) and Pi-Sigma networks (Shin & Ghosh, 1991). #link("https://arxiv.org/abs/1905.12207")[Kileel et al. (2019)] analyzed functional varieties of deep polynomial networks, while #link("https://arxiv.org/abs/2006.13026")[Chrysos et al. (2021)] scaled high-order tensor-factorized $Pi$-Nets (`CCP`, `NCP`, `NCP-Skip`)—usually inserting normalization or `tanh`. #link("https://arxiv.org/abs/2405.07200")[Sidharth et al. (2024)] introduced `ChebyKAN` using `tanh`-clamped Chebyshev bases; our fixed-point CAS formulation eliminates `tanh` entirely. On sets, #link("https://arxiv.org/abs/1703.06114")[Zaheer et al. (2017)] and #link("https://arxiv.org/abs/1901.09006")[Wagstaff et al. (2019)] characterized power-sum symmetric encoders, and #link("https://arxiv.org/abs/1810.00825")[Lee et al. (2019)] introduced `SetTransformer`. Differentiable sorting relaxations—`NeuralSort` (#link("https://arxiv.org/abs/1903.08850")[Grover et al., 2019]), Sinkhorn OT (#link("https://arxiv.org/abs/1905.11885")[Cuturi et al., 2019]), permutahedron projection (#link("https://arxiv.org/abs/2002.08871")[Blondel et al., 2020]), `SoftSort` (#link("https://arxiv.org/abs/2006.16038")[Prillo & Eisenschlos, 2020]), and `diffsort` (#link("https://arxiv.org/abs/2105.04019")[Petersen et al., 2021]\; #link("https://arxiv.org/abs/2203.09630")[Petersen et al., 2022])—all require non-polynomial operations. Conversely, homomorphic encryption (CKKS) sorting (#link("https://eprint.iacr.org/2019/1234")[Cheon et al., 2020]; Hong et al., 2021; #link("https://eprint.iacr.org/2020/834")[Lee et al., 2022]\; #link("https://arxiv.org/abs/2412.15126")[Mazzone et al., 2025]) operates strictly in the polynomial ring but relies on fixed analytical coefficients.

*Implications for Encrypted Inference (CKKS) and Circuit Design.* Our results have an immediate practical implication for CKKS encrypted sorting: *training composite comparator polynomials end-to-end on the target data distribution cuts value sorting error by 1.6×–3.9× at the exact same multiplicative depth and bootstrapping budget!* Furthermore, our A2 ablation identifies `L=4, d=3` (fixed-point nested cubics) as a superior accuracy–stability operating point ($R_"stable" = 1.50$, `91.8%` Ex\@0.05 at $n=8$) compared to quintic `L=3, m=2` ($R_"stable" = 1.00$), and A4 identifies Batcher Odd-Even Merge as strictly better than Bitonic at identical stage count.

*Limitations.* (1) *Fixed input scale domain:* Pure polynomials require inputs within their stability radius (typically $s <= 1.0$–$1.5$); unbounded inputs require external rescaling (e.g., centered non-poly scaling `(x - mean)/max(|x - mean|)` to preserve both shift and scale equivariance). (2) *Multiplicative depth scaling:* `PolySortNet` has sequential depth $O(L log^2 n)$, whereas `PolyPairwiseRank` has constant depth $O(1)$ but loses fidelity at $n >= 16$ unless rank-indicator degree scales with $n$. (3) *Power-of-2 wiring:* Bitonic/Odd-Even Merge natively sort power-of-2 lengths (arbitrary $n$ requires padding with $+1$ sentinels). (4) *Single-seed ablations:* Stages 4D and 4E-sens ran on seed `42` (mitigated by verifying near-zero `PolySortNet` seed variance `SD <= 8e-5` across all 5 seeds in Stage 4E). (5) *Non-polynomial ceiling:* When strict polynomiality is unnecessary (standard plaintext ML), `DiffSort` and `SetTransformer` achieve 2.3×–14× lower in-distribution MAE.

= Conclusion

We presented the first preregistered, mechanically purity-audited scientific study of strictly pure Polynomial Neural Networks for continuous list sorting across 178 experimental runs. Algorithmic inductive bias—combining data-oblivious sorting-network wiring with deep composite odd Chebyshev comparators—bridges nearly the entire accuracy gap between unstructured high-order PNNs and transcendental neural sorters while using *3–60 parameters*, outperforming hand-crafted CKKS minimax polynomials by up to *3.9×* at iso-depth, extrapolating *16× zero-shot to $n=128$*, and beating a 96.6k-parameter `SetTransformer` on Shift $+100$, Near-Ties, Duplicates, and Zero-Shot $n=128$. Equally importantly, our causal ablations honestly falsified two preregistered expectations—uncovering the late-stage domain-contraction trade-off behind fixed-point anchoring and the sharp separation between iso-degree and iso-parameter polynomial depth—establishing a grounded blueprint for polynomial circuit learning.

= References

#set text(size: 8.3pt)
- Batcher, K. E. (1968). Sorting Networks and Their Applications. _AFIPS Spring Joint Computer Conference_, 307–314.
- Bernstein, S. (1914). Sur la meilleure approximation de $|x|$ par des polynômes de degrés donnés. _Acta Mathematica_, 37(1), 1–57.
- Blondel, M., Teboul, O., Berthet, Q., & Djolonga, J. (2020). Fast Differentiable Sorting and Ranking. _ICML 2020_, #link("https://arxiv.org/abs/2002.08871")[arXiv:2002.08871].
- Cheon, J. H., Kim, D., & Kim, D. (2020). Efficient Homomorphic Comparison Methods with Optimal Complexity. _ASIACRYPT 2020_, #link("https://eprint.iacr.org/2019/1234")[IACR ePrint 2019/1234].
- Chrysos, G. G., Moschoglou, S., Bouritsas, G., Panagakis, Y., Deng, J., & Zafeiriou, S. (2021). Deep Polynomial Neural Networks. _IEEE TPAMI_, #link("https://arxiv.org/abs/2006.13026")[arXiv:2006.13026] (#link("https://github.com/grigorisg9gr/polynomial_nets")[Code]).
- Cuturi, M., Teboul, O., & Vert, J.-P. (2019). Differentiable Ranks and Sorting using Optimal Transport. _NeurIPS 2019_, #link("https://arxiv.org/abs/1905.11885")[arXiv:1905.11885].
- Grover, A., Wang, E., Zweig, A., & Ermon, S. (2019). Stochastic Optimization of Sorting Networks via Continuous Relaxations. _ICLR 2019_, #link("https://arxiv.org/abs/1903.08850")[arXiv:1903.08850].
- Habermann, A. N. (1972). Parallel Neighbor-Sort. _Carnegie-Mellon University Technical Report_.
- Hong, S., Kim, S., Choi, J., Lee, Y., & Cheon, J. H. (2021). Efficient Sorting of Homomorphic Encrypted Data with $k$-Way Sorting Network. _IEEE TIFS_, 16, 4389–4404.
- Ivakhnenko, A. G. (1971). Polynomial Theory of Complex Systems. _IEEE Trans. Systems, Man, and Cybernetics_, 1(4), 364–378.
- Kileel, J., Trager, M., & Bruna, J. (2019). On the Expressive Power of Deep Polynomial Neural Networks. _NeurIPS 2019_, #link("https://arxiv.org/abs/1905.12207")[arXiv:1905.12207].
- Knuth, D. E. (1998). _The Art of Computer Programming, Vol. 3: Sorting and Searching_ (2nd ed.), Sec. 5.3.4. Addison-Wesley.
- Lee, E., Lee, J.-W., No, J.-S., & Kim, Y.-S. (2022). Minimax Approximation of Sign Function by Composite Polynomial for Homomorphic Comparison. _IEEE TDSC_, 19(6), 3711–3727, #link("https://eprint.iacr.org/2020/834")[IACR ePrint 2020/834].
- Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019). Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. _ICML 2019_, #link("https://arxiv.org/abs/1810.00825")[arXiv:1810.00825].
- Lipton, Z. C., & Steinhardt, J. (2018). Troubling Trends in Machine Learning Scholarship. #link("https://arxiv.org/abs/1807.03341")[arXiv:1807.03341].
- Lones, M. A. (2021/2024). How to Avoid Machine Learning Pitfalls: A Guide for Academic Researchers. #link("https://arxiv.org/abs/2108.02497")[arXiv:2108.02497].
- Mazzone, F., Everts, M., Hahn, F., & Peter, A. (2025). Efficient Ranking, Order Statistics, and Sorting under CKKS. _USENIX Security 2025_, #link("https://arxiv.org/abs/2412.15126")[arXiv:2412.15126].
- Nanda, N. (2022–2024). How I Think About My Research Process. #link("https://www.lesswrong.com/s/5GT3yoYM9gRmMEKqL")[LessWrong / Alignment Forum Sequence].
- Petersen, F., Borgelt, C., Kuehne, H., & Deussen, O. (2021). Differentiable Sorting Networks for Scalable Sorting and Ranking Supervision. _ICML 2021_, #link("https://arxiv.org/abs/2105.04019")[arXiv:2105.04019] (#link("https://github.com/Felix-Petersen/diffsort")[Code]).
- Petersen, F., Borgelt, C., Kuehne, H., & Deussen, O. (2022). Monotonic Differentiable Sorting Networks. _ICLR 2022_, #link("https://arxiv.org/abs/2203.09630")[arXiv:2203.09630].
- Prillo, S., & Eisenschlos, J. (2020). SoftSort: A Continuous Relaxation for the argsort Operator. _ICML 2020_, #link("https://arxiv.org/abs/2006.16038")[arXiv:2006.16038].
- Shin, Y., & Ghosh, J. (1991). The Pi-Sigma Network: An Efficient Higher-Order Neural Network. _IJCNN 1991_, 1, 13–18.
- Sidharth, S. S., et al. (2024). Chebyshev Polynomial-Based Kolmogorov-Arnold Networks. #link("https://arxiv.org/abs/2405.07200")[arXiv:2405.07200].
- Varga, R. S., & Carpenter, A. J. (1985). On the Bernstein Conjecture in Approximation Theory. _Numerische Mathematik_, 47, 179–196.
- Wagstaff, E., Fuchs, F. B., Engelcke, M., Posner, I., & Osborne, M. A. (2019). On the Limitations of Representing Functions on Sets. _ICML 2019_, #link("https://arxiv.org/abs/1901.09006")[arXiv:1901.09006].
- Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R., & Smola, A. (2017). Deep Sets. _NeurIPS 2017_, #link("https://arxiv.org/abs/1703.06114")[arXiv:1703.06114].

#pagebreak()
#set text(size: 9pt)
#heading(numbering: none)[Appendix A: Exact Learned Pure-Polynomial Coefficients]

For complete reproducibility (and direct drop-in replacement inside CKKS encrypted sorting circuits), the exact learned fixed-point odd Chebyshev layer coefficients $h_k (u) = u + u(1 - u^2) sum_(j=0)^(m-1) c_(k,j) T_(2j)(u)$ for our 3-parameter and 6-parameter zero-shot shared networks (`seed=42`, trained on $n=8$) are:
- *`PolySortNet-Shared (m=1)` (3 scalars, deg 27, zero-shot $n=128$ MAE `0.0581`):*
  $c_1 = [1.4017459]$, $c_2 = [0.8520995]$, $c_3 = [0.7529720]$.
- *`PolySortNet-Shared (m=2)` (6 scalars, deg 125, zero-shot $n=128$ MAE `0.0200`, $tau = 0.9978$):*
  $c_1 = [0.7872190, -1.5390211]$, $c_2 = [1.1491840, -0.9942412]$, $c_3 = [0.9384390, -0.3469870]$.

Notice that both models autonomously learn *decreasing higher-order gain* across composite layers (strong initial amplification followed by gentle terminal contraction). All 8 pure-polynomial browser models execute identically (`0.00e+00` float64 max abs diff vs. PyTorch) live at #link("https://burnycoder.github.io/polynomial-neural-network-sort-list/")[burnycoder.github.io/polynomial-neural-network-sort-list].

#heading(numbering: none)[Appendix B: Hyperparameter, Batch-Size, and `float64` Sensitivity Audit (Stage 4E-sens)]

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, left, right, right, right, right, right),
    [Sensitivity Axis], [Configuration ($n=8$, seed 42)], [ID MAE], [RMSE], [Mono %], [Ex\@0.05], [$R_"stab"$],
    [PolySort LR], [`lr=3e-4` / `1e-3` / `3e-3` / `8e-3` / `1e-2`], [0.0570 / 0.0384 / 0.0338 / 0.0333 / *0.0332*], [0.071..0.042], [87–94%], [0.7–16.6%], [0.0 / 1.3],
    [PiNet LR Audit], [`lr=3e-4` / `1e-3` / `3e-3` / `1e-2`], [0.0771 / 0.0745 / 0.0520 / *0.0423*], [0.099..0.056], [91–97%], [0.4–7.7%], [0.0 / 1.1],
    [Weight Decay], [`wd=0.0` vs. `wd=0.01` (`PolySortNet`)], [0.033254 vs. 0.033258], [0.04191], [94.09% / 94.02%], [16.4%], [1.30],
    [Batch Size], [`bs=512` / `2048` / `4096` (40 ep)], [*0.03165* / 0.03321 / 0.03327], [0.0397..0.0419], [94.1–94.3%], [20.8–16.4%], [1.2 / 1.3],
    [Float64 Train], [`L=3` (`float64` vs. `float32`)], [*0.033254* vs. 0.033254 (6-dec match)], [0.041907], [94.09%], [16.38%], [1.30],
    [Float64 Train], [`L=4` deg 81 (`float64` vs. `float32`)], [*0.013222* vs. 0.013222 (6-dec match)], [0.017113], [96.22%], [91.80%], [1.50],
  ),
  caption: [*Stage 4E sensitivity sweep (15 runs).* Confirms zero `float32` numerical drift even at degree 81, weight-decay invariance, slightly conservative `bs=4096` throughput setting, and honest LR tuning of `PiNet-NCP-Skip`.],
)

#heading(numbering: none)[Appendix C: NeurIPS Paper Checklist Verification]
- *Claims & Limitations:* Every claim is backed by paired statistical tests, 95% CIs, or algebraic proof; both falsified preregistered hypotheses (H3.1, H6.1) and pure-polynomial scale blow-up limitations are prominently disclosed.
- *Code, Data, Checkpoints & Reproducibility:* Full modular `src/pnn_sort/` package, 62 automated `pytest` tests (including FX purity & Knuth 0-1 verification), all 178 run JSONs, and trained checkpoints are publicly archived on GitHub and Hugging Face with deterministic SHA-256 split seeds.
- *Statistical Rigor:* 5 independent seeds across all 10 core models $times$ 2 lengths (100 cells), 10,000-resample bootstrap CIs, paired Wilcoxon signed-rank tests with rank-biserial effect sizes, and Holm-Bonferroni correction across all 34 tests.
- *Compute Accounting:* Single NVIDIA RTX 5070 Laptop GPU (8 GB VRAM), $approx 5$ hours total compute across 178 runs; all training scripts support atomic resume.
