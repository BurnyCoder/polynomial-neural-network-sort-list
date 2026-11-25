# Cumulative Scientific & Engineering Learnings (`pnn-sort`)

**Author:** Libor "Burny" Burian  
**Last Updated:** 2026-11-25  

This living log documents every key insight gained from the literature review, mathematical derivations, engineering/debugging mistakes, and empirical experiments.

---

## Learnings from Literature & State-of-the-Art Review (Phase 1)

1. **Unifying Three Siloed Fields Reveals the Core Opportunity:**
   - **Deep PNNs ($\Pi$-Nets; Chrysos et al., TPAMI 2021)** study expressive multivariate polynomial approximations (CCP, NCP, NCP-Skip, ProdPoly) but almost always insert BatchNorm/InstanceNorm or `tanh` to prevent high-order gradient blow-up, and have not been studied on non-smooth algorithmic sorting.
   - **Differentiable Sorting Networks (`diffsort`; Petersen et al., ICML 2021 / ICLR 2022)** prove exact algebraic properties of odd-symmetric conditional swaps (sum conservation, shift invariance, idempotency, boundedness) and an $\epsilon \cdot \ell$ depth error bound (Theorem 9), plus monotonicity conditions ($f'(x) \in \Theta(1/x^2)$, Theorems 5 & 8) — but rely strictly on non-polynomial transcendental functions ($\exp$, $\arctan$, division).
   - **Homomorphic CKKS Polynomial Sorting (Cheon et al., 2020; Hong et al., 2021; Lee et al., 2022; Mazzone et al., USENIX Security 2025)** solves sorting with **strictly pure polynomials**, introducing composite sign polynomials $f_m(u)$, Remez minimax composite sequences, and Mazzone's constant-depth all-to-all ranking with exact quadratic tie-correction $E = 4C(1-C)$ — but uses fixed, hand-engineered coefficients minimizing **unweighted sign error** $|1 - s(u)|$ instead of data-trained **value-weighted sorting error** $|\Delta|\,|1 - s(|\Delta|)|$.

2. **Key Mathematical Derivation — Fixed-Point Odd Chebyshev Parameterization:**
   - Parameterizing each composite layer as $h_k(u) = u + u(1 - u^2)\sum_{j=0}^{m-1} c_{k,j} T_{2j}(u)$ strictly guarantees $h_k(0)=0, h_k(\pm 1)=\pm 1$, odd symmetry, and exact subsumption of both the identity map ($c=0$) and Cheon et al.'s analytical $f_m(u)$ polynomials (verified symbolically to $< 10^{-13}$ in SymPy).
   - Computing even Chebyshev terms via $T_{2j}(u) = T_j(2u^2 - 1)$ halves the multiplicative steps while remaining 100% pure polynomial.
   - Because $\Delta = (b - a)/2 \in [-1, 1]$ for $a, b \in [-1, 1]$, any comparator satisfying $|s(\Delta)| \le 1$ keeps every wire bounded inside $[\min(x), \max(x)] \subseteq [-1, 1]$ across arbitrary sorting depth (preventing activation blow-up without any non-polynomial normalization!).

3. **Rolle's Theorem Trade-Off Between Exact Boundary Fidelity and Monotonicity:**
   - Checking Petersen et al. (2022) Theorem 5 against Cheon's $f_1(u) = \frac{3}{2}u - \frac{1}{2}u^3$ revealed a subtle mathematical fact: any smooth comparator with $s(0)=0, s(1)=1$, and $s(u)<1$ on $(0,1)$ has $\hat{\min}_s(2, 0) = 0$ and $\hat{\min}_s(x, 0) > 0$ on $(0, 2)$, so by Rolle's theorem its derivative $\frac{d}{dx}\hat{\min}_s(x, 0)$ MUST become negative somewhere on $(0, 2)$. Thus exact boundary accuracy at $u=1$ precludes strict global monotonicity on $[0, 1]$, while Petersen's Theorem 9 ($\epsilon \cdot \ell$ error bound) still holds directly!

---

## Learnings from Pre-Training Engineering & Verification (Phase 3)

4. **Bug Caught by Resume Determinism Test — Cosine LR Horizon Must Remain Fixed:**
   - *Issue:* Our initial `test_checkpoint_resume_determinism` failed (`0.5277` vs `0.5203`) because passing `epochs=2` then `epochs=4` created two different cosine schedule horizons (`total_steps`), causing epochs 1–2 to use different learning rates than an uninterrupted 4-epoch run.
   - *Fix:* Added `stop_after_epoch` to `train_run` so the target horizon `cfg.epochs` stays invariant across interruptions and resumes bit-for-bit.

5. **Mathematical Insight Caught by Analytical Unit Test — Why Repeated Cheon $f_1^{(3)}$ Has High MAE ($\approx 0.114$) and Why Mixed/Learned Compositions Win:**
   - *Discovery:* Our initial unit test expected Cheon $f_1^{(3)}$ to achieve MAE $< 0.02$ on $n=8$ Uniform $[-1, 1]$, but its actual exact MAE was `0.1137`!
   - *Root Cause:* Because $f_1(u) = \frac{3}{2}u - \frac{1}{2}u^3$ has origin slope $f_1'(0) = 1.5$, composing it 3 times only yields origin slope $1.5^3 = 3.375$. For $n=8$ lists in $[-1, 1]$, typical adjacent half-differences are $\Delta \approx 0.11$, where $f_1^{(3)}(0.11) \approx 0.36$ — far from $1.0$!
   - *Scientific Upgrade:* To avoid comparing against a weak analytical strawman, we added **Cheon/Hong/Mazzone's exact mixed $f_3 \circ g_3$** polynomial (origin slope $4.48 \times 2.19 \approx 9.8$, degree 49, MAE $< 0.03$) AND **Remez minimax composite polynomials** at BOTH $m=1$ (deg 27) and $m=2$ (deg 125). This also explains *mechanistically* why gradient-trained composite polynomials (`c_1 > c_2 > c_3`) can crush uniform $f_1^{(3)}$ at the exact same degree 27: early layers can amplify small gaps aggressively (controlled intermediate overshoot, just like Cheon's $g_3$), while the final layer contracts values cleanly back to $\pm 1$!

6. **Lee et al. (2022) Exact Convex-LP Composite Remez & Stability Threshold Discovery:**
   - *Issue:* Our initial Nelder-Mead fixed-point minimax approximation at $\delta = 0.02$ produced poor MAE (`0.312`), and even Lee et al.'s exact closed-form odd cubic minimax at $\delta = 0.02$ ($\tau_L = 0.526$) diverged to `NaN` inside a 6-stage Bitonic network!
   - *Root Cause:* When $\delta$ is set too aggressively for composition depth $L = 3$, the minimax equioscillation amplitude $\tau_L$ pushes $|s(\Delta)|$ up to $1 + \tau_L \approx 1.53$, allowing wire values to escape $[-1, 1]$ and cascade exponentially across sorting stages! Conversely, scaling the final stage by $1/(1+\tau_L)$ keeps wires bounded but shrinks every swap by $(1+\tau_L)^{-1}$ per stage, compounding to $(1+\tau_L)^{-\ell}$ attenuation.
   - *Fix:* Implemented **Lee et al. (2022) Algorithm 2** via exact **HiGHS convex linear programming** (`scipy.optimize.linprog`), rescaling intermediate stages $1 \dots L-1$ by $1/\text{peak}$ into $[\alpha_k, 1]$ while centering stage $L$ at $1.0$ with depth-matched $\delta$ ($\delta=0.10$ for $L=3, m=1 \Rightarrow \text{MAE} \approx 0.044$, $2.6\times$ better than Cheon $f_1^{(3)}$; $\delta=0.03$ for $L=3, m=2$).

---

## Learnings from Stage 4A Execution & GPU Profiling

7. **Bug Fixed in `R_stable` Upper Stability Radius Calculation:**
   - *Issue:* During Stage 4A controls, `ana_cheon_fg_n8` achieved `MAE = 0.0279` at $s = 1.0$, yet reported `R_stable = 0.0`.
   - *Root Cause:* `SCALE_SWEEP_FACTORS` begins with `0.1, 0.5, 1.0, 1.05, ...`. At tiny shrinkage $s = 0.1$, all pairwise differences shrink by $10\times$ into the near-origin linear zone of $s(\Delta)$, making relative MAE (`MAE / 0.1`) $> 0.05$ and tripping `break` before ever evaluating $s \ge 1.0$.
   - *Fix:* Updated `metrics.py` so `R_stable` starts at $s = 1.0$ and sweeps upward ($s \ge 1.0$), cleanly measuring the upper polynomial blow-up cliff while still logging full $s = 0.1, 0.5$ metrics in `ood`.

8. **8× RTX 5070 Throughput Fix (Windows WDDM Kernel-Launch Bottleneck):**
   - *Observation:* `main_polysort_shared_n8_s42` at `batch_size=512` took `~10.5s/epoch` despite converging to `val_mae = 0.03776` by epoch 8 (99% GPU idle due to ~150 tiny CUDA kernel launches $\times$ 391 batches/epoch under Windows WDDM).
   - *Fix:* Switched compact structured networks (`PolySortNet`, `PolyPairwiseRank`, `DiffSort`) to `batch_size=4096, lr=8e-3, epochs=50` (~49 batches/epoch, ~8× faster) and dense networks to `batch_size=2048, lr=2e-3, epochs=60`, while retaining `batch_size=512 vs 2048 vs 4096` explicitly in the Stage 4E sensitivity sweep to verify invariance.



---

## Learnings from Stage 4C & Stage 4D (Baselines & Causal Ablations)

9. **Unconstrained Polynomials Exploit Late-Stage Domain Contraction (H6 Part 1 Falsified on ID MAE):**
   - Freeing $h_k(1) \neq 1$ (`unconstrained_odd_cheby`, `raw_monomial`, `general_cheby`) cuts ID MAE by ~26–30% (`0.0234 / 0.0273` vs `0.0333 / 0.0369`) because late Bitonic stages only receive small $|\Delta| \lesssim 0.4$; unconstrained polynomials contract their valid domain to sharpen origin slope (`9.58 -> 14.3`), letting `s_5(1.0)` diverge to `-444..-1038` and shrinking `R_stable` from `1.3 -> 1.1`. Fixed-point anchoring is thus a **scale-stability regularizer**, not an ID-MAE maximizer.

10. **Iso-Degree vs. Iso-Parameter Separation & Optimal Depth-vs-Width Frontier (H3 Refined):**
    - Flat degree 9 (`L=1, d=9`, 4 params/stage) beats composite degree 9 (`L=2, d=3`, 2 params/stage) on ID MAE (`0.0566` vs `0.0771`), falsifying H3 Part 1 at strict iso-degree $D=9$, yet exhibits Runge boundary blow-up (`R_stable = 0.00`). At **equal parameter budget (4 params/stage)**, composite **`L=4, d=3` (degree 81)** achieves `$4.28\times$` lower MAE (`0.0132 / 0.0163`), `91.8% / 61.4%` Ex@0.05, AND widens `R_stable` to **`1.50`** — strictly superior in stability to quintic `L=3, m=2` (`R_stable = 1.00`).

11. **Shift vs. Scale Symmetry Trade-Off in Non-Polynomial Normalization (A6):**
    - Wrapping `PolySortNet` in `x / max(|x|)` lifts `R_stable` from `1.30 -> 5.00` (proving H5 causally), while breaking Shift $+100$ equivariance (`0.0334 -> 0.4553`) because uncentered scaling compresses pairwise differences by $101\times$.

12. **Topology-Dependent Reversal Asymmetry (A4):**
    - Batcher Odd-Even Merge improves MAE by 10–13% over Bitonic and preserves exact sorted/reversed symmetry (`EdgeS == EdgeR`), while Habermann Odd-Even Transposition achieves the highest monotonicity (`99.30% / 96.77%`) at the cost of $2.74\times$ higher error on reversed inputs at $n=16$ (`0.1349` vs `0.0493`).

---

## Learnings from Stage 4E (Sensitivity Sweeps & 5-Seed Statistical Rigor)

13. **Zero `float32` Optimization Drift Even at Degree 81:**
    - Full `float64` training matched `float32` to 6 decimal places on both $L=3$ (`0.033254`) and $L=4$ (`0.013222`), confirming fixed-point odd Chebyshev bounds protect single-precision gradients.
14. **Four Orders of Magnitude Lower Seed Variance & Fair Baseline LR Audit:**
    - Across 5 seeds (`42, 123, 456, 789, 1024`), 36-param `PolySortNet-Adaptive (m=2)` exhibits seed SD `5.8e-7` (`n=8`) vs `5.2e-3` (`PiNet`) and `2.9e-3` (`PolyMLP`). Baseline LR tuning lifted `PiNet` from `0.0529 -> 0.0423` at `lr=0.01` (reported honestly; still $5.85\times$ behind `PolySortNet-Adaptive m=2`).
15. **Mechanistic Bitonic Stride-Reset Rediscovery (Visual Audit of Fig. 3C):**
    - Visually inspecting our stage-adaptive slope plot revealed that gradient descent autonomously drops comparator origin slope $s'(0)$ at every fresh long-stride Bitonic merge boundary (Stage 3 for $n=8$; Stages 1, 3, 6 for $n=16$) and ramps slope monotonically as stride halves `8 -> 4 -> 2 -> 1`.
