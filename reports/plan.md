# Master Project Plan (`pnn-sort`)

**Author:** Libor "Burny" Burian  
**Created:** 2026-11-25  
**Last Updated:** 2026-11-25  

## Objective
Train, rigorously evaluate, ablate, visualize, and publish a scientific study of **strictly pure Polynomial Neural Networks (PNNs)** for sorting continuous lists of real numbers, comparing unstructured PNNs ($\Pi$-Nets, PolyMLP), symmetric set PNNs (DeepSets-Poly), constant-depth all-to-all polynomial rankers (`PolyPairwiseRank`), and learned composite polynomial sorting networks (`PolySortNet`) against zero-training analytical CKKS minimax polynomials (Cheon et al., 2020; Lee et al., 2022), trivial/random controls, and non-polynomial neural baselines (`diffsort`, ReLU-MLP, Set Transformer).

## Phase Checklist
- [x] **Phase 0:** Workspace & hardware audit (NVIDIA RTX 5070 Laptop GPU, CUDA 13.4, `torch 2.11.0+cu128`, `uv`, `gh`, `hf`, `pdflatex`, `typst`).
- [x] **Phase 1:** Full primary-source literature review (`reports/literature-review.md`) and SOTA review (`reports/state-of-the-art-review.md`).
- [x] **Phase 2:** Preregistration (`reports/preregistration.md`), Run Matrix (`reports/planned-runs.md`), Docs (`docs/`), Pre-Results `README.md`, GitHub + Hugging Face repo creation & pre-results push.
- [x] **Phase 3:** Modular `src/pnn_sort/` package, automated FX purity audit, real-time accuracy web dashboard, full `pytest` suite (62/62 passed), pre-training push & Adversarial Self-Review #1.
- [x] **Phase 4A:** Zero-training analytical/controls + primary `PolySortNet` (Shared & Stage-Adaptive), incremental GitHub & HF pushes.
- [x] **Phase 4B:** Remaining main PNN architectures (`PolyPairwiseRank`, `PolySortNetLearnedWire`, `DeepSetsPoly`, `PiNetSorter`, `PolyMLP`) + scaling study ($n=4..128$), Review #2.
- [x] **Phase 4C:** Trained controls (`ShuffledTargetControl`) & non-polynomial baselines (`DiffSort`, `ReLU-MLP`, `SetTransformer`).
- [x] **Phase 4D:** 6 causal ablation suites (30 runs) + Adversarial NeurIPS Self-Review #3.
- [x] **Phase 4E:** 5-seed statistical rigor (80 new runs; 100 core cells) + hyperparameter/batch/float64 sensitivity sweep (15 runs) — **178/178 runs DONE**.
- [x] **Phase 5:** Publication visualizations, GitHub Pages live browser polynomial sorter, full scientific paper (`reports/paper.pdf`, `reports/summary.md`), Final Triple-Check & Adversarial Review #4, final GitHub + HF sync — **100% COMPLETE**.
