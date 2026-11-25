# Scientific Methodology & Reproducibility Guide

## 1. Pure-Polynomial Verification (`src/pnn_sort/audit.py`)
Every model declaring `is_pure_polynomial = True` is symbolically traced via `torch.fx.symbolic_trace`. The auditor inspects every IR node and fails the build if ANY operation outside `{add, sub, mul, matmul, linear, sum, einsum, permute, transpose, reshape, view, unsqueeze, squeeze, expand, cat, stack, slice, getitem, neg, pow(int >= 0)}` is present. Specifically, `relu`, `gelu`, `sigmoid`, `tanh`, `softmax`, `atan`, `exp`, `log`, `div` (by tensor), `abs`, `sqrt`, `max`, `min`, `clamp`, `sort`, `argsort`, and normalization layers are strictly forbidden in pure PNN models.

## 2. Reproducibility & Resumability
- Every run records full config JSON, Git commit SHA, PyTorch/CUDA version, hardware info, and deterministic seeds.
- Checkpoints are written atomically (`*.tmp -> *.pt`) with model weights, optimizer, LR scheduler, epoch counter, best validation metric, and Python/NumPy/PyTorch/CUDA RNG states so interrupted runs resume bit-for-bit cleanly.
- All raw per-run and per-test-condition outputs are saved under `results/` as machine-readable JSON/CSV.

## 3. Statistical Rigor
- Core models are trained across 5 independent random seeds (`42, 123, 456, 789, 1024`).
- Uncertainty is quantified via both inter-seed `mean ± std` and 10,000-resample paired bootstrap 95% confidence intervals on $N_{\text{test}} = 10{,}000$ lists.
- Formal model comparisons use two-sided paired Wilcoxon signed-rank tests with Holm-Bonferroni correction across preregistered hypotheses H1–H6.
