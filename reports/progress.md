# Live Progress Report (`pnn-sort`)

**Last Updated:** 2026-11-25 17:10 CET  

## Current Status
- **Overall Progress:** ~48% complete (Phases 0–3 & Stage 4A complete; launching Stage 4B).
- **Done:**
  - Phases 0–3: Environment, full literature & SOTA reviews, pre-results preregistration, modular `src/pnn_sort/` codebase, 62/62 passing tests, Review #1.
  - **Stage 4A Complete (26 runs):** All zero-training analytical/trivial controls + all 8 primary `PolySortNet` models (`shared`, `shared_m2`, `adaptive`, `adaptive_m2` at $n=8, 16$).
    - **Positive Highlights:**
      - `polysort_adaptive_m2` achieves **MAE = `0.00724`** ($n=8$, 36 params, Mono `95.92%`, $\tau=0.9967$) and **MAE = `0.00973`** ($n=16$, 60 params, $\tau=0.9963$) — **$3.85\times$ lower MAE** than Cheon/Hong/Mazzone mixed $f_3 \circ g_3$.
      - `polysort_shared_m2` (**6 params**, trained on $n=8$ only) generalizes **zero-shot to $n=128$** with **MAE = `0.0200`, $\tau = 0.9978$** ($13\times$ better than Cheon $f_1^{(3)}$).
      - Exact float64 sum conservation (`2.1e-16`) and shift equivariance at $c = 100$ (`6.6e-14`).
    - **Negative / Boundary Highlights:**
      - Analytical unscaled Remez $m=2$ diverged (`MAE = inf`) across deep Bitonic stages due to compounding overshoot.
      - Pure polynomials exhibit the exact predicted OOD scale-cliff: cubic ($m=1$) diverges at $s = 2.0$ ($R_{\text{stable}} = 1.3$); quintic ($m=2$) diverges at $s = 1.5$ ($R_{\text{stable}} = 1.0$).
- **In Progress:** Pushing Stage 4A artifacts & checkpoints to GitHub and Hugging Face, then running **Stage 4B** (17 runs: `PolyPairwiseRank`, `LearnedWire`, `DeepSetsPoly`, `PiNet-NCPSkip`, `PolyMLP` on $n=8, 16$, plus $n=4, 32$ scaling sweep).
- **Estimated Remaining Time:** ~1.5–2.5 hours total for Stages 4B–4E + Phase 5 (figures, GitHub Pages live browser sorter, `paper.pdf`).
