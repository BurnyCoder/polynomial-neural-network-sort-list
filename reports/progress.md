# Live Progress Report (`pnn-sort`)

**Last Updated:** 2026-11-25 17:52 CET  

## Current Status
- **Overall Progress:** ~66% complete (Phases 0–3, Stages 4A–4C complete; launching Stage 4D causal ablations).
- **Done:**
  - Stages 4A, 4B, and 4C complete (53 runs total).
  - Stage 4C Highlights: Zero data leakage confirmed (`ShuffledTarget` MAE `0.210`/`0.282`); 6-param pure-polynomial `PolySortNet-Shared-m2` beats 37k-param `ReLU-MLP` by $5.7\times$ ($n=16$) and beats 96.6k-param `SetTransformer` on Shift $+100$ ($12{,}500\times$), Zero-Shot $n=128$ ($2.65\times$), and Near-Ties $10^{-3}$ (`0.00666` vs `0.00757`), while honestly lagging non-polynomial `DiffSort` (`exp`/`atan`) on ID MAE and scale extrapolation ($s=5$).
- **In Progress:** Pushing Stage 4C to GitHub & Hugging Face, then running **Stage 4D** (30 causal ablation runs across A1 Parameterization, A2 Iso-Degree Depth Separation, A3 Initialization, A4 Wiring + Random-Wire Negative Control, A5 Loss Function, and A6 Scale-Norm Non-Poly Boundary; ~25–35 min on RTX 5070).
- **Next Up:** Adversarial Self-Review #3, Stage 4E (5-seed statistical rigor + sensitivity sweeps), and Phase 5 (figures, GitHub Pages live browser sorter, `reports/paper.pdf`).
