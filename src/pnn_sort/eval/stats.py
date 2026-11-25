"""Statistical rigor utilities: bootstrap 95% CIs, paired Wilcoxon tests, Holm-Bonferroni correction."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon


def bootstrap_ci(
    values: np.ndarray,
    num_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 2026,
) -> Tuple[float, float, float]:
    """Return ``(mean, ci_low, ci_high)`` via percentile bootstrap."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = arr.size
    # Vectorized chunked bootstrap to keep memory low
    chunk = 1000
    means = []
    for start in range(0, num_resamples, chunk):
        b = min(chunk, num_resamples - start)
        idx = rng.integers(0, n, size=(b, n))
        means.append(arr[idx].mean(axis=1))
    boot = np.concatenate(means)
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return float(arr.mean()), lo, hi


def paired_wilcoxon_test(
    a: np.ndarray,
    b: np.ndarray,
    num_boot: int = 10_000,
    seed: int = 2026,
) -> Dict[str, float]:
    """Two-sided paired Wilcoxon signed-rank test on ``diff = a - b`` with bootstrap CI & effect size."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    diff = arr_a - arr_b
    finite = np.isfinite(diff)
    diff = diff[finite]
    mean_diff, ci_lo, ci_hi = bootstrap_ci(diff, num_resamples=num_boot, seed=seed)

    if diff.size == 0 or np.allclose(diff, 0.0):
        return {
            "mean_diff": mean_diff,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "p_value": 1.0,
            "statistic": 0.0,
            "rank_biserial": 0.0,
            "n_pairs": int(diff.size),
        }

    res = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
    # Rank-biserial correlation r = (W+ - W-) / (W+ + W-)
    nonzero = diff[diff != 0.0]
    n_nz = nonzero.size
    total_rank_sum = n_nz * (n_nz + 1) / 2.0
    pos_ranks = np.argsort(np.argsort(np.abs(nonzero))) + 1
    w_plus = float(pos_ranks[nonzero > 0].sum())
    w_minus = total_rank_sum - w_plus
    rb = (w_plus - w_minus) / total_rank_sum if total_rank_sum > 0 else 0.0

    return {
        "mean_diff": mean_diff,
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "p_value": float(res.pvalue),
        "statistic": float(res.statistic),
        "rank_biserial": float(rb),
        "n_pairs": int(diff.size),
    }


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Compute Holm-Bonferroni familywise-adjusted p-values preserving original order."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda kv: kv[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, p * (m - rank))
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max
    return adjusted
