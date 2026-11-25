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

