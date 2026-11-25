# Experiment Stage 4D: Six Causal Ablation Suites (A1–A6)

**Author:** Libor "Burny" Burian  
**Date:** 2026-11-25  
**Runs Covered:** 30 controlled causal ablations (15 interventions $\times$ $n \in \{8, 16\}$, seed `42`, $N_{\text{test}} = 10{,}000$) compared against the reference pure-polynomial model `main_polysort_adaptive` ($L=3, m=1$, fixed-point odd Chebyshev, Bitonic wiring, Cheon init, MSE loss).  
**Primary Purpose:** Test preregistered hypotheses **H3** (iso-degree / iso-parameter composite depth separation), **H5** (causal proof of the polynomial scale-explosion boundary via A6), and **H6** (parameterization constraints and sorting-network wiring topology).

---

## 1. Plain-Language Summary (For Non-Experts)

To prove *why* our Polynomial Sorting Network (`PolySortNet`) works — and where it breaks — we changed **one design ingredient at a time** while holding everything else constant. We found **two honest falsifications** of our initial hypotheses and **four strong causal confirmations**:

1. **Allowing polynomial comparators to drift away from $s(1)=1$ (A1) actually *improves* in-domain accuracy by ~26–30%**, falsifying H6 Part 1 — *because* late stages of a Bitonic network only see tiny value gaps, so unconstrained polynomials shrink their safe range to boost sharpness. **The catch:** their transfer curve explodes to $-444\text{ to }-1038$ at $u=1.0$, shrinking the safe scale radius from $1.3\times$ down to $1.1\times$.
2. **At identical degree 9 (A2), a flat 4-parameter polynomial beats a 2-parameter 2-layer composite polynomial**, falsifying H3 Part 1 at strict iso-degree $D=9$. **However, at identical parameter count (4 params/stage)**, composing 4 cubics (`L=4, d=3`, degree 81) crushes the flat degree-9 polynomial by **$4.28\times$** in MAE (`0.0132` vs `0.0566`), boosts exact-sort rate from **`1.4%` $\to$ `91.8%`**, and *expands* scale stability from `0.0` (Runge boundary blow-up at $1.1\times$) to **`1.50`**.
3. **Wiring topology is causally indispensable (A4):** Replacing valid sorting-network wiring with random fixed pairs destroys sorting completely (`0.00%` monotonicity, MAE `0.386–0.436`), while switching from Bitonic to **Batcher Odd-Even Merge** cuts MAE by 10–13%, and **Habermann Odd-Even Transposition** pushes monotonicity up to **`99.30%`** ($n=8$) and **`96.77%`** ($n=16$).
4. **Adding a single non-polynomial `max(|x|)` division wrapper (A6)** immediately eliminates the polynomial scale blow-up (`R_stable: 1.3 -> 5.0`), proving causally that pure polynomial unboundedness causes the scale cliff — *while revealing an inverse trade-off*: uncentered max-abs scaling breaks Shift $+100$ equivariance (`0.033 -> 0.455`), which pure `PolySortNet` preserves to float64 machine precision.

---

## 2. Complete Quantitative Results Table (Seed 42, $N_{\text{test}} = 10{,}000$)

### 2.1 List Length $n = 8$ (6 Bitonic / Odd-Even Merge stages; 8 Odd-Even Transposition stages)

| Ablation Group | Run ID | Pure Poly? | Params | ID MAE ($\downarrow$) | ID RMSE ($\downarrow$) | Mono% ($\uparrow$) | Ex@0.05% ($\uparrow$) | Kendall $\tau$ ($\uparrow$) | Near-Ties $10^{-3}$ | Shift $+100$ | Scale $1.5\times$ | $R_{\text{stable}}$ ($\uparrow$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference** | `main_polysort_adaptive_n8` | Yes | 18 | `0.03325` | `0.04191` | `94.09%` | `16.38%` | `0.9949` | `0.03267` | `0.03341` | `0.16553` | `1.30` |
| **A1 Param** | `abl_param_unconst_odd_n8` | Yes | 36 | **`0.02342`** | `0.03005` | `93.10%` | **`48.13%`** | `0.9942` | `0.02399` | `0.02335` | `inf` | `1.10` |
| **A1 Param** | `abl_param_general_cheby_n8` | Yes | 72 | **`0.02341`** | `0.02991` | `92.23%` | `47.63%` | `0.9936` | `0.02396` | `0.02333` | `inf` | `1.10` |
| **A1 Param** | `abl_param_raw_monomial_n8` | Yes | 36 | `0.02381` | `0.03040` | `93.56%` | `47.07%` | `0.9945` | `0.02424` | `0.02373` | `inf` | `1.10` |
| **A2 Depth** | `abl_depth_L1_d9_n8` (flat deg 9) | Yes | 24 | `0.05662` | `0.07065` | `97.81%` | `1.37%` | `0.9973` | `0.05840` | `0.05702` | `inf` | `0.00` |
| **A2 Depth** | `abl_depth_L2_d3_n8` (comp deg 9) | Yes | 12 | `0.07712` | `0.09711` | `90.33%` | `0.15%` | `0.9904` | `0.07868` | `0.07691` | `0.11705` | `0.00` |
| **A2 Depth** | `abl_depth_L4_d3_n8` (comp deg 81) | Yes | 24 | **`0.01322`** | **`0.01711`** | `96.22%` | **`91.80%`** | `0.9969` | **`0.01261`** | **`0.01328`** | **`0.06191`** | **`1.50`** |
| **A3 Init** | `abl_init_identity_n8` ($c=0$) | Yes | 18 | `0.03478` | `0.04381` | `91.30%` | `12.96%` | `0.9921` | `0.03490` | `0.03516` | `0.08844` | `1.30` |
| **A3 Init** | `abl_init_random_n8` | Yes | 18 | `0.03437` | `0.04333` | `92.74%` | `14.05%` | `0.9934` | `0.03476` | `0.03470` | `0.10715` | `1.30` |
| **A4 Wire** | `abl_wire_oddeven_merge_n8` | Yes | 18 | **`0.02884`** | `0.03637` | `93.53%` | **`25.75%`** | `0.9944` | `0.03328` | `0.02855` | `0.08379` | `1.30` |
| **A4 Wire** | `abl_wire_oddeven_trans_n8` | Yes | 24 | `0.03830` | `0.04794` | **`99.30%`** | `7.88%` | **`0.9988`** | `0.04592` | `0.03838` | `inf` | `1.30` |
| **A4 Wire** | `abl_wire_random_fixed_n8` (neg ctrl) | Yes | 18 | `0.38611` | `0.49144` | `0.00%` | `0.00%` | `0.3386` | `0.34673` | `0.38716` | `inf` | `0.00` |
| **A5 Loss** | `abl_loss_huber_n8` | Yes | 18 | `0.03219` | `0.04126` | `93.34%` | `20.16%` | `0.9943` | `0.03216` | `0.03217` | `0.16914` | `1.30` |
| **A5 Loss** | `abl_loss_mse_mono_n8` ($\lambda=0.1$) | Yes | 18 | `0.03327` | `0.04191` | `94.63%` | `16.63%` | `0.9954` | `0.03264` | `0.03341` | `0.16772` | `1.30` |
| **A5 Loss** | `abl_loss_mse_mono_w1_n8` | Yes | 18 | `0.03229` | `0.04088` | `93.84%` | `19.70%` | `0.9947` | `0.03221` | `0.03230` | `0.16835` | `1.30` |
| **A6 Bound** | `abl_scale_norm_n8` (non-poly) | **No** | 18 | `0.02934` | `0.03726` | `94.86%` | `27.78%` | `0.9955` | `0.02321` | **`0.45534`** | **`0.04386`** | **`5.00`** |

### 2.2 List Length $n = 16$ (10 Bitonic / Odd-Even Merge stages; 16 Odd-Even Transposition stages)

| Ablation Group | Run ID | Pure Poly? | Params | ID MAE ($\downarrow$) | ID RMSE ($\downarrow$) | Mono% ($\uparrow$) | Ex@0.05% ($\uparrow$) | Kendall $\tau$ ($\uparrow$) | Near-Ties $10^{-3}$ | Shift $+100$ | Scale $1.5\times$ | $R_{\text{stable}}$ ($\uparrow$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference** | `main_polysort_adaptive_n16` | Yes | 30 | `0.03694` | `0.04688` | `75.81%` | `0.95%` | `0.9942` | `0.03904` | `0.03699` | `0.15316` | `1.30` |
| **A1 Param** | `abl_param_unconst_odd_n16` | Yes | 60 | `0.02783` | `0.03538` | `84.66%` | `9.21%` | `0.9966` | `0.02840` | `0.02782` | `inf` | `1.10` |
| **A1 Param** | `abl_param_general_cheby_n16` | Yes | 120 | `0.02809` | `0.03576` | `82.36%` | `8.32%` | `0.9962` | `0.02864` | `0.02800` | `inf` | `1.10` |
| **A1 Param** | `abl_param_raw_monomial_n16` | Yes | 60 | **`0.02731`** | `0.03500` | `88.16%` | **`11.23%`** | `0.9974` | `0.02794` | `0.02730` | `inf` | `1.10` |
| **A2 Depth** | `abl_depth_L1_d9_n16` (flat deg 9) | Yes | 40 | `0.06742` | `0.08513` | `90.41%` | `0.00%` | `0.9980` | `0.07241` | `0.06759` | `inf` | `0.00` |
| **A2 Depth** | `abl_depth_L2_d3_n16` (comp deg 9) | Yes | 20 | `0.08312` | `0.10570` | `90.26%` | `0.00%` | `0.9974` | `0.08587` | `0.08360` | `0.14470` | `0.00` |
| **A2 Depth** | `abl_depth_L4_d3_n16` (comp deg 81) | Yes | 40 | **`0.01631`** | **`0.02081`** | `81.47%` | **`61.38%`** | `0.9960` | **`0.01691`** | **`0.01633`** | **`0.06218`** | **`1.50`** |
| **A3 Init** | `abl_init_identity_n16` ($c=0$) | Yes | 30 | `0.03891` | `0.04931` | `73.33%` | `0.58%` | `0.9934` | `0.04105` | `0.03904` | `0.11562` | `1.30` |
| **A3 Init** | `abl_init_random_n16` | Yes | 30 | `0.03993` | `0.05053` | `67.31%` | `0.52%` | `0.9913` | `0.04124` | `0.04011` | `0.10396` | `1.30` |
| **A4 Wire** | `abl_wire_oddeven_merge_n16` | Yes | 30 | **`0.03326`** | `0.04242` | `80.85%` | `2.29%` | `0.9953` | `0.04078` | `0.03327` | `0.08392` | `1.30` |
| **A4 Wire** | `abl_wire_oddeven_trans_n16` | Yes | 48 | `0.04114` | `0.05240` | **`96.77%`** | `0.41%` | **`0.9983`** | `0.04999` | `0.04118` | `inf` | `1.30` |
| **A4 Wire** | `abl_wire_random_fixed_n16` (neg ctrl) | Yes | 30 | `0.43629` | `0.51605` | `0.00%` | `0.00%` | `0.2229` | `0.39606` | `0.43794` | `inf` | `0.00` |
| **A5 Loss** | `abl_loss_huber_n16` | Yes | 30 | `0.03637` | `0.04761` | `73.11%` | `1.40%` | `0.9935` | `0.03935` | `0.03654` | `0.16244` | `1.30` |
| **A5 Loss** | `abl_loss_mse_mono_n16` ($\lambda=0.1$) | Yes | 30 | `0.03830` | `0.04838` | **`82.52%`** | `0.73%` | `0.9952` | `0.03827` | `0.03833` | `0.15539` | `1.30` |
| **A5 Loss** | `abl_loss_mse_mono_w1_n16` | Yes | 30 | `0.03667` | `0.04687` | `75.23%` | `1.09%` | `0.9940` | `0.03908` | `0.03674` | `0.15353` | `1.30` |
| **A6 Bound** | `abl_scale_norm_n16` (non-poly) | **No** | 30 | `0.03512` | `0.04463` | `76.34%` | `1.56%` | `0.9944` | `0.03314` | **`0.47950`** | **`0.05278`** | **`5.00`** |

---

## 3. Mechanistic Gate Inspection (Direct Weight Transfer Evaluation in Float64)

Evaluating the learned Stage-0 and Final-Stage (Stage 5) comparator polynomials on $n=8$ directly in float64 explains *every* ablation effect causally:

| Model ($n=8$, seed 42) | St0 $s(0)$ | St0 $s'(0)$ | St0 $s(1.0)$ | St0 $s(1.3)$ | St5 $s'(0)$ | St5 $s(1.0)$ | St5 $s(1.3)$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `main_polysort_adaptive` (`fixed_point_cheby`, $L=3$) | `+0.0000` | `6.78` | `1.0000` | `0.874` | `9.58` | `1.0000` | `0.013` |
| `abl_param_unconst_odd` (A1) | `+0.0000` | `9.35` | `0.7949` | `-15.811` | `14.22` | **`-444.67`** | **`-2.30e8`** |
| `abl_param_raw_monomial` (A1) | `+0.0000` | `8.41` | `0.8101` | `0.967` | `14.33` | **`-1037.69`** | **`-3.47e8`** |
| `abl_param_general_cheby` (A1) | **`-0.0117`** | `9.30` | `0.7859` | `-14.667` | `14.25` | **`-680.69`** | **`-2.73e8`** |
| `abl_depth_L1_d9` (flat deg 9, A2) | `+0.0000` | `5.20` | `1.0000` | `22.231` | `5.23` | `1.0000` | `56.988` |
| `abl_depth_L2_d3` (comp deg 9, A2) | `+0.0000` | `3.93` | `1.0000` | `0.815` | `5.07` | `1.0000` | `-0.985` |
| `abl_depth_L4_d3` (comp deg 81, A2) | `+0.0000` | **`13.45`** | `1.0000` | `0.626` | **`20.04`** | `1.0000` | `-0.998` |

### Key Causal Takeaways
1. **A1 (Domain Contraction in Late Stages — H6 Part 1 Falsified on MAE, Refined as Accuracy-vs-$R_{\text{stable}}$ Pareto Frontier):** Removing $h_k(1)=1$ frees late stages to contract their operating domain to $|\Delta| \lesssim 0.5$ (where Stage 5 inputs live after Stages 0–4), lifting Stage-5 slope $s'(0)$ from `9.58 -> 14.2–14.3` and improving ID + in-domain OOD MAE (`Bimodal` improves $2.2\times$ at $n=16$: `0.0389` vs `0.0858`), while letting $s_5(1.0)$ diverge to `-444..-1038` and causing immediate blow-up at `Scale 1.3x` (`R_stable: 1.3 -> 1.1`). Removing odd symmetry (`general_cheby`) adds $s(0) = -0.0117 \neq 0$, slightly breaking sorted/reversed symmetry (`0.02962` vs `0.02967`) with zero accuracy gain over odd unconstrained Chebyshev.
2. **A2 (Iso-Degree Falsification vs. Iso-Parameter Separation — H3 Refined):** At strict iso-degree $D=9$, the 2-parameter composite manifold (`L=2, d=3`) is strictly contained in the 4-parameter flat degree-9 polynomial space (`L=1, d=9`), so flat deg-9 beats `L=2, d=3` on ID MAE (`0.0566` vs `0.0771` at $n=8$). However, at **equal parameter budget (4 params/stage)**, `L=4, d=3` (degree 81) beats flat deg-9 by **$4.28\times$ ($n=8$) / $4.13\times$ ($n=16$)**, reaches **`91.8%` / `61.4%` Ex@0.05** (vs `1.4%` / `0.0%`), and avoids flat degree-9's Runge boundary explosion ($s(1.1) = 1.92 \Rightarrow R_{\text{stable}} = 0.00$), expanding stability all the way to **`R_stable = 1.50`** (`|s(1.5)| <= 1.005`) while retaining float64 exact sum conservation (`2.24e-16`) and shift equivariance (`7.75e-14`)!
3. **A3 (Initialization):** Cheon init outperforms identity ($c=0$) and random init by 4.4–8.1% in MAE (`0.0333` vs `0.0348`/`0.0344` at $n=8$; `0.0369` vs `0.0389`/`0.0399` at $n=16$), with **0% divergence** across all three initializations thanks to fixed-point odd Chebyshev stability.
4. **A4 (Wiring Topology — H6 Part 2 Confirmed):** Valid sorting-network wiring is causally necessary (`random_fixed` collapses to `0.00%` Mono, $\tau=0.22–0.34$). Batcher Odd-Even Merge improves MAE (`-13.3%` at $n=8$, `-9.9%` at $n=16$) and halves Scale $1.5\times$ error (`0.0839` vs `0.1532`) by executing fewer redundant comparisons than Bitonic on inner subarrays. Habermann Odd-Even Transposition trades higher MAE (and reversed-input asymmetry: `EdgeR = 0.1349` vs `EdgeS = 0.0493` at $n=16$) for near-perfect local ordering (`99.30%` / `96.77%` Mono, $\tau = 0.9988 / 0.9983$).
5. **A5 (Loss Function):** Huber slightly reduces MAE (`-3.2%` at $n=8$, `-1.5%` at $n=16$); MSE + Monotonicity hinge (`w=0.1`) lifts $n=16$ monotonicity by **+6.71 percentage points** (`75.81% -> 82.52%`) at a modest `+3.7%` MAE cost (`0.0369 -> 0.0383`).
6. **A6 (Non-Polynomial Scale-Norm Boundary — H5 Causal Proof):** Adding `max(|x|)` division before the exact same inner polynomial increases $R_{\text{stable}}$ from `1.30 -> 5.00` (`Scale 5.0x` MAE `0.146 / 0.176`, exact scale proportionality `MAE/s = const`), proving causally that pure polynomial unboundedness drives the scale cliff — while breaking Shift $+100$ equivariance (`0.455 / 0.480`) because uncentered scaling compresses shifted differences by $101\times$.
