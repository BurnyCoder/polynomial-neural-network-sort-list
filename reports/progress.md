# Live Progress Report (`pnn-sort`)

**Last Updated:** 2026-11-25 17:36 CET  

## Current Status
- **Overall Progress:** ~58% complete (Phases 0–3, Stages 4A & 4B, and Review #2 complete; launching Stage 4C).
- **Done:**
  - Stages 4A & 4B complete (43 runs total): full pure-polynomial architecture ladder (`PolySortNet`, `PolyPairwiseRank`, `PolySortNetLearnedWire`, `DeepSetsPoly`, `PiNet-NCPSkip`, `PolyMLP`), all zero-training analytical CKKS baselines, trivial/linear controls, scaling sweep ($n=4..32$ + zero-shot $n=4..128$), and Adversarial Self-Review #2 (`reports/reviews/review-02-main-methods.md`).
- **In Progress:**
  - Pushing Stage 4B artifacts, Review #2, and checkpoints to GitHub & Hugging Face, then running **Stage 4C** (10 runs: `ShuffledTargetControl`, `DiffSort-Logistic`, `DiffSort-Cauchy`, `ReLU-MLP`, `SetTransformer` across $n=8, 16$, ~8–10 min on RTX 5070).
- **Next Up:** Stage 4D (30 causal ablation runs across A1–A6) + Review #3, Stage 4E (5-seed statistical rigor + sensitivity sweeps), and Phase 5 (figures, GitHub Pages live browser sorter, `reports/paper.pdf`).
- **Estimated Remaining Time:** ~1.2–2.0 hours wall-clock.
