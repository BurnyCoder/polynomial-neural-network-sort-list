# Live Progress Report (`pnn-sort`)

**Last Updated:** 2026-11-25 18:50 CET  

## Current Status
- **Overall Plan Completion:** **~74% complete** (**~26% remaining**; **83 / 178** preregistered experiment runs complete).
- **Live Accuracy Monitor:** Active at `http://127.0.0.1:18080`.
- **Done:**
  - Phases 0–3 (Lit/SOTA review, Preregistration `H1–H6`, Modular `src/pnn_sort/`, 62/62 pytests, Review #1).
  - Stages 4A (26 runs), 4B (17 runs + Review #2), 4C (10 runs), and **Stage 4D (30/30 causal ablations A1–A6 + Adversarial Review #3)** — all pushed to GitHub & Hugging Face.
- **In Progress Now:**
  - Pushing Stage 4D + Review #3 to GitHub & Hugging Face, then executing **Stage 4E** (`4e_sens`: 15 hyperparameter/batch-size/float64 runs, followed by `4e_seeds`: 80 multi-seed replication runs across seeds `123, 456, 789, 1024`; ~75–95 min on RTX 5070).
- **Next Up (Phase 5, ~45 min):**
  - 5-seed statistical analysis (10k bootstrap 95% CIs, paired Wilcoxon + Holm-Bonferroni), publication figures, live interactive GitHub Pages browser pure-polynomial sorter (`docs/index.html`), full academic paper `reports/paper.pdf` (by Libor "Burny" Burian) + `reports/summary.md`, comprehensive `README.md`, Final Triple-Check & Adversarial Review #4, final GitHub & HF sync.
