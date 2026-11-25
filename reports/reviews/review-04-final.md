# Adversarial Self-Review #4: Final Pre-Publication Scientific & Reproducibility Audit

**Date:** 2026-11-25  
**Reviewer Role:** Skeptical NeurIPS Area Chair & Methodology Auditor  
**Scope:** End-to-end triple-check across all **178 / 178 runs**, 5-seed statistics (100 core cells), 34 Holm-Bonferroni Wilcoxon tests, 62 automated `pytest` tests, all 6 publication figures, live browser pure-polynomial evaluator (`docs/index.html`), `reports/paper.pdf`, `reports/summary.md`, and `README.md`.

---

## 1. Scientific & Statistical Integrity Checklist (NeurIPS Standard)

| Audit Criterion | Status | Concrete Ground-Truth Verification |
| :--- | :---: | :--- |
| **Zero Data Leakage / Contamination** | **PASS** | SHA-256 keyed disjoint splits verified in `tests/test_data.py`; `ShuffledTarget` leakage control confirmed (`0.210 / 0.282` MAE, `0.0%` Ex@0.05). |
| **Strict Pure-Polynomial Enforcement** | **PASS** | Automated PyTorch FX tracer (`src/pnn_sort/audit.py`) audited 18/18 pure-PNN models and rejected 5/5 non-poly models (`62/62` pytest tests passing after final `trainer.py` edit). |
| **No p-Hacking / Pre- vs. Post-Hoc Transparency** | **PASS** | Preregistration (`H1–H6`) pushed to GitHub (`879258c`) *before* any training/eval results existed; all post-hoc adjustments (`analytical_cheon_fg`, HiGHS LP Remez, `bs=4096` WDDM fix, `R_stable` $s \ge 1.0$) explicitly logged in `preregistration.md` Sec. 7. |
| **Honest Reporting of Negative & Falsified Results** | **PASS** | Both falsified preregistered hypotheses (**H3.1** iso-degree $D=9$ and **H6.1** fixed-point ID MAE), the cubic $m=1$ vs. `DeepSets-Poly` $n=8$ tie (**H1d**), fair LR tuning of `PiNet-NCP-Skip` (`0.0423` at `lr=0.01`), pure-polynomial scale blow-up cliffs, and non-poly upper-bound gaps (`DiffSort`, `SetTransformer`) are prominently featured in Abstract, Tables 1–3, Fig. 4, Summary, and README. |
| **Multi-Seed Error Bars, CIs & Multiple-Testing Control** | **PASS** | All 10 core models $\times$ $n \in \{8, 16\}$ replicated across **5 independent seeds** (`42, 123, 456, 789, 1024`; 100 cells, 50,000 test lists/model). All 34 planned comparisons verified via two-sided paired Wilcoxon signed-rank tests, 10,000-resample bootstrap 95% CIs, rank-biserial effect sizes, and Holm-Bonferroni correction (`results/statistical_tests.json`). |
| **Confounder & Numerical Precision Audits** | **PASS** | Stage 4E sensitivity sweep confirmed zero `float32` vs. `float64` optimization drift to 6 decimal places even at degree $3^4 = 81$ (`L=4`), weight-decay invariance (`< 4e-6`), and conservative `bs=4096` setting (`0.03327` vs `bs=512` `0.03165`). |
| **No Graph Crimes & Visual PDF Audit** | **PASS** | Every PNG figure (`fig1..fig6`) and every page of `reports/paper.pdf` visually inspected directly; caught and fixed overlapping A1 scatter labels in Fig. 6B, `RandomFixed` scale-2x badge in Fig. 4, Table 1 header spacing, Typst math subscripts/fractions, and literal `~` spacing. Colorblind-safe Okabe-Ito palette throughout. |
| **Numerical Consistency Across Paper, README, & Raw JSONs** | **PASS** | Automated triple-check verified `178/178` run JSONs, `20/20` 5-seed cells, `34` Holm-Bonferroni tests, `6/6` figures in both `reports/paper/figures/` and `docs/figures/`, and `19/19` verbatim numerical cross-checks across `paper.typ` and `README.md`. |
| **Bit-Exact Browser Deployment & Reproducibility** | **PASS** | All 8 exported pure-polynomial models in `docs/weights.json` verified bit-exact (`0.00e+00` max abs diff vs. PyTorch float64) via Node.js; GitHub Pages live at `https://burnycoder.github.io/polynomial-neural-network-sort-list/`. |

---

## 2. Residual Limitations (Explicitly Retained in Paper Section 7)
1. Stage-4D causal ablations and Stage-4E sensitivity runs are single-seed (`seed=42`); mitigated by verifying that `PolySortNet` exhibits seed SD $\le 8 \times 10^{-5}$ across all 5 seeds in Stage 4E.
2. Pure polynomials cannot eliminate upper scale divergence ($s > R_{\text{stable}}$) without leaving the polynomial ring; centered non-polynomial scaling `(x - mean)/max(|x - mean|)` is left as clear future work.
3. Bitonic/Odd-Even Merge wiring requires power-of-2 lengths natively (arbitrary lengths require $+1$ sentinel padding).

---

## 3. Final Publication Verdict
**ACCEPT (Camera-Ready).** All methodological, statistical, reproducibility, visualization, and scientific-integrity standards are met without outstanding defects. Approved for final synchronization to GitHub (`main` + GitHub Pages) and Hugging Face Hub.
