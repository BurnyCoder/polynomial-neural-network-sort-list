# Live Progress Report (`pnn-sort`)

**Last Updated:** 2026-11-25 16:40 CET  

## Current Status
- **Overall Progress:** ~38% complete (Phases 0–3 complete; starting Phase 4A training).
- **Done:**
  - Phase 0–1: CUDA GPU environment (`torch 2.11.0+cu128`, RTX 5070 Laptop GPU), full primary-source literature & SOTA reviews.
  - Phase 2: Preregistration (`reports/preregistration.md`), Run Matrix (`reports/planned-runs.md`), docs, GitHub + Hugging Face repos created, pre-results commit `879258c` pushed.
  - Phase 3: Modular `src/pnn_sort/` codebase implemented (17+ models/controls/baselines, PyTorch FX purity auditor, exact HiGHS convex-LP Lee et al. 2022 Remez composite solver, resumable trainer, real-time web accuracy dashboard, bootstrap/Wilcoxon/Holm stats, plotting engine). **62/62 pytest tests pass**. Adversarial Self-Review #1 written (`reports/reviews/review-01-pre-training.md`).
- **In Progress:**
  - Pushing verified Phase 3 codebase to GitHub & Hugging Face, then launching Stage 4A (zero-training analytical/controls + primary `PolySortNet` models on $n=8, 16$) with live dashboard monitoring.
- **Next Up:** Stages 4B (full PNN ladder + scaling $n=4..128$), 4C (non-poly baselines + shuffled-target control), 4D (6 causal ablations), 4E (5-seed statistical rigor + sensitivity sweeps), and Phase 5 (visualizations, GitHub Pages live browser sorter, `reports/paper.pdf`).
- **Estimated Remaining Time:** ~1.5–3 hours wall-clock on RTX 5070.
