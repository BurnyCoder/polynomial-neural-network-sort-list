# Live Progress Report (`pnn-sort`)

**Last Updated:** 2026-11-25 19:17 CET  

## Current Status
- **Overall Plan Completion:** **~78% complete** (**~22% remaining**; **98 / 178** preregistered experiment runs complete).
- **Live Accuracy Monitor:** Active at `http://127.0.0.1:18080`.
- **Done:**
  - Phases 0–3 (Lit/SOTA review, Preregistration `H1–H6`, Modular `src/pnn_sort/`, 62/62 pytests, Review #1).
  - Stages 4A (26 runs), 4B (17 runs + Review #2), 4C (10 runs), Stage 4D (30 causal ablations + Review #3).
  - **Stage 4E Sensitivity Sweeps (`4e_sens`, 15/15 runs DONE):**
    - `float64` training matches `float32` to **6 decimal places** on both `L=3` (`0.033254`) and degree-81 `L=4` (`0.013222`), ruling out precision drift.
    - Weight-decay invariant (`wd=0`: `0.033254` vs `wd=0.01`: `0.033258`); batch-size audit confirms `bs=4096` (`0.03327`) is slightly conservative vs `bs=512` (`0.03165`).
    - Fair LR audit shows `PiNet-NCP-Skip` improves to `0.04232` at `lr=0.01` (still $5.85\times$ behind 36-param `PolySortNet-Adaptive-m2`).
  - Phase 5 scaffolding built & verified early: `aggregate.py`, `build_paper_assets.py` (6 polished figures), and interactive browser pure-polynomial sorter `docs/index.html` (verified `0.00e+00` bit-exact float64 parity vs PyTorch across all 8 exported models!).
- **In Progress Now:**
  - Pushing `4e_sens` + Phase 5 web/viz scaffolding to GitHub & Hugging Face, then executing **`4e_seeds`** (**80 runs**: 10 core models $\times$ $n \in \{8, 16\}$ $\times$ 4 remaining seeds `123, 456, 789, 1024`, with mid-stage GitHub/HF push after the 40 $n=8$ runs; ~80–100 min total on RTX 5070).
- **Next Up (Final Phase 5, ~35–45 min):**
  - Re-run `aggregate.py` & `build_paper_assets.py` on all 178 runs (final 5-seed `mean ± std`, 10k bootstrap 95% CIs, Holm-Bonferroni Wilcoxon tests), write `stage-4e` report, compile full academic paper `reports/paper.pdf` (by Libor "Burny" Burian) + `reports/summary.md`, update comprehensive `README.md`, enable live GitHub Pages (`docs/`), complete **Final Triple-Check & Adversarial Review #4**, final GitHub & HF push.
