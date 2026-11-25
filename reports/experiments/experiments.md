# Master Experiments Log (`pnn-sort`)

**Author:** Libor "Burny" Burian  
**Last Updated:** 2026-11-25  

## Plain-Language Glossary of Key Terms
- **Pure Polynomial Neural Network (PNN):** A neural network whose forward calculation uses *only* additions, subtractions, and multiplications — no `ReLU`, `sigmoid`, `softmax`, `if/else`, divisions, or built-in `sort`.
- **Composite Polynomial Depth ($L$):** Instead of fitting one giant flat polynomial (which needs huge degree to approximate sharp corners like $|a-b|$), we feed a simple cubic or quintic polynomial into itself $L$ times. Three nested cubics produce a degree-$3^3 = 27$ polynomial with just 3 learnable numbers!
- **Sorting Network Wiring (Bitonic / Odd-Even):** A fixed blueprint of which pairs of slots to compare at each parallel step. Replacing each discrete compare-and-swap switch with a pure polynomial yields a **Polynomial Sorting Network (`PolySortNet`)**.
- **Value MAE ($\downarrow$):** Average absolute difference between the network's output list and the true sorted list. Closer to `0.0000` is better.
- **Monotonicity Rate ($\uparrow$):** Percentage of test lists where every output value is in valid sorted order ($\hat{y}_1 \le \hat{y}_2 \le \dots \le \hat{y}_n$).
- **Pairwise Order Accuracy / Kendall's $\tau$ ($\uparrow$):** Percentage of all element pairs in the output list that are in the correct relative order ($\tau = 1.0$ means zero inversions).
- **Scale Stability Radius ($R_{\text{stable}}$):** How much larger than $[-1, 1]$ input values can grow before a pure polynomial's higher-order terms inevitably explode.

---

## Summary Table of Completed Experiments

*(Populated incrementally as Stages 4A → 4E execute.)*

| Experiment Folder | Stage | Models Covered | Status | Key Highlight |
| :--- | :---: | :--- | :---: | :--- |
| `stage-4a-controls-and-main` | 4A | Analytical CKKS, Trivial/Linear Controls, `PolySortNet` (Shared & Stage-Adaptive) | **Done** | 6-param `polysort_shared_m2` gets `0.0079` MAE ($n=8$), `0.0200` MAE zero-shot at $n=128$ ($\tau=0.998$); 36-param `adaptive_m2` gets `0.0072` MAE ($n=8$), `0.0097` ($n=16$) — beating Cheon $f_3\circ g_3$ by $3.85\times$! |
| `stage-4b-pnn-ladder-and-scaling` | 4B | `PolyPairwiseRank`, `LearnedWire`, `DeepSetsPoly`, `PiNet`, `PolyMLP`, $n=4..128$ | **Done** | Full pure-PNN ladder complete: `polysort_adaptive_m2` reaches `82.0%` Ex@0.05 at $n=16$ vs `0.7%` (`DeepSetsPoly`) and `0.0%` (`PiNet`/`PolyMLP`/`Pairwise`); stage-adaptive gain grows monotonically `-6.2% -> -31.5%` from $n=4 \to 32$. |
| `stage-4c-baselines-and-controls` | 4C | `ShuffledTarget`, `DiffSort` (Logistic/Cauchy), `ReLU-MLP`, `SetTransformer` | **Done** | Zero leakage verified (`ShuffledTarget` MAE `0.21–0.28`); `PolySortNet-m2` beats `ReLU-MLP` by $6.5\times$ ($n=16$) and beats 96.6k-param `SetTransformer` on Shift $+100$ ($12{,}500\times$), Zero-Shot $n=128$ ($2.65\times$), and Near-Ties $10^{-3}$; `DiffSort` leads ID & Scale $5\times$ via transcendental $\exp/\arctan$. |
| `stage-4d-ablations` | 4D | 6 Causal Ablation Families (A1–A6) | In Progress | — |
| `stage-4e-multiseed-and-sensitivity` | 4E | 5-Seed Statistical Rigor (100 runs) + Hyperparam & Float64 Sweeps | Pending | — |
