"""Aggregate multi-seed summaries, 95% bootstrap CIs, and Holm-Bonferroni Wilcoxon tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from pnn_sort.config import DEFAULT_SEEDS, RESULTS_DIR
from pnn_sort.eval.stats import bootstrap_ci, holm_bonferroni, paired_wilcoxon_test


CORE_PREFIXES = [
    ("main_polysort_shared", "PolySortNet-Shared (m=1)", "main_pnn"),
    ("main_polysort_adaptive", "PolySortNet-Adaptive (m=1)", "main_pnn"),
    ("main_polysort_adaptive_m2", "PolySortNet-Adaptive (m=2)", "main_pnn"),
    ("main_poly_pairwise", "PolyPairwiseRank (Mazzone)", "main_pnn"),
    ("main_deepsets_poly", "DeepSets-Poly", "main_pnn"),
    ("main_pinet_ncpskip", "PiNet-NCP-Skip", "main_pnn"),
    ("main_poly_mlp", "PolyMLP (x^2+x)", "main_pnn"),
    ("base_diffsort_cauchy", "DiffSort-Cauchy (non-poly)", "non_poly"),
    ("base_relu_mlp", "ReLU/GELU-MLP (non-poly)", "non_poly"),
    ("base_set_transformer", "SetTransformer (non-poly)", "non_poly"),
]

# Preregistered paired comparisons (A vs B: negative mean_diff means A has LOWER MAE = better)
PREREGISTERED_TESTS: List[Tuple[str, str, str, str, int, bool]] = [
    # (test_id, hypothesis, prefix_a, prefix_b, n, use_multiseed)
    ("H1a_n8", "H1: PolySortNet-Adaptive-m2 vs DeepSets-Poly (n=8)", "main_polysort_adaptive_m2", "main_deepsets_poly", 8, True),
    ("H1a_n16", "H1: PolySortNet-Adaptive-m2 vs DeepSets-Poly (n=16)", "main_polysort_adaptive_m2", "main_deepsets_poly", 16, True),
    ("H1b_n8", "H1: PolySortNet-Adaptive-m2 vs PiNet-NCP-Skip (n=8)", "main_polysort_adaptive_m2", "main_pinet_ncpskip", 8, True),
    ("H1b_n16", "H1: PolySortNet-Adaptive-m2 vs PiNet-NCP-Skip (n=16)", "main_polysort_adaptive_m2", "main_pinet_ncpskip", 16, True),
    ("H1c_n8", "H1: PolySortNet-Adaptive-m2 vs PolyMLP (n=8)", "main_polysort_adaptive_m2", "main_poly_mlp", 8, True),
    ("H1c_n16", "H1: PolySortNet-Adaptive-m2 vs PolyMLP (n=16)", "main_polysort_adaptive_m2", "main_poly_mlp", 16, True),
    ("H1d_n8", "H1 (cubic caveat): PolySortNet-Adaptive-m1 vs DeepSets-Poly (n=8)", "main_polysort_adaptive", "main_deepsets_poly", 8, True),
    ("H1d_n16", "H1 (cubic): PolySortNet-Adaptive-m1 vs DeepSets-Poly (n=16)", "main_polysort_adaptive", "main_deepsets_poly", 16, True),
    ("H2a_n8", "H2: PolySortNet-Adaptive-m1 vs Analytical Cheon f1^(3) (n=8)", "main_polysort_adaptive", "ana_cheon_f1_L3", 8, False),
    ("H2a_n16", "H2: PolySortNet-Adaptive-m1 vs Analytical Cheon f1^(3) (n=16)", "main_polysort_adaptive", "ana_cheon_f1_L3", 16, False),
    ("H2b_n8", "H2: PolySortNet-Adaptive-m1 vs Analytical Remez L=3 (n=8)", "main_polysort_adaptive", "ana_remez_L3", 8, False),
    ("H2b_n16", "H2: PolySortNet-Adaptive-m1 vs Analytical Remez L=3 (n=16)", "main_polysort_adaptive", "ana_remez_L3", 16, False),
    ("H2c_n8", "H2: PolySortNet-Adaptive-m2 vs Analytical Cheon f3∘g3 (n=8)", "main_polysort_adaptive_m2", "ana_cheon_fg", 8, False),
    ("H2c_n16", "H2: PolySortNet-Adaptive-m2 vs Analytical Cheon f3∘g3 (n=16)", "main_polysort_adaptive_m2", "ana_cheon_fg", 16, False),
    ("H2d_n8", "H2 (stage-adaptive): Adaptive-m1 vs Shared-m1 (n=8)", "main_polysort_adaptive", "main_polysort_shared", 8, True),
    ("H2d_n16", "H2 (stage-adaptive): Adaptive-m1 vs Shared-m1 (n=16)", "main_polysort_adaptive", "main_polysort_shared", 16, True),
    ("H3a_n8", "H3 (iso-degree D=9): Comp L=2,d=3 vs Flat L=1,d=9 (n=8)", "abl_depth_L2_d3", "abl_depth_L1_d9", 8, False),
    ("H3a_n16", "H3 (iso-degree D=9): Comp L=2,d=3 vs Flat L=1,d=9 (n=16)", "abl_depth_L2_d3", "abl_depth_L1_d9", 16, False),
    ("H3b_n8", "H3 (iso-param 4/stage): Comp L=4,d=3 vs Flat L=1,d=9 (n=8)", "abl_depth_L4_d3", "abl_depth_L1_d9", 8, False),
    ("H3b_n16", "H3 (iso-param 4/stage): Comp L=4,d=3 vs Flat L=1,d=9 (n=16)", "abl_depth_L4_d3", "abl_depth_L1_d9", 16, False),
    ("H4_n8", "H4: PolySortNet-Adaptive-m2 vs PolyPairwiseRank (n=8)", "main_polysort_adaptive_m2", "main_poly_pairwise", 8, True),
    ("H4_n16", "H4: PolySortNet-Adaptive-m2 vs PolyPairwiseRank (n=16)", "main_polysort_adaptive_m2", "main_poly_pairwise", 16, True),
    ("H6a_n8", "H6 (fixed-point): Fixed-Point vs Unconstrained Odd Cheby (n=8)", "main_polysort_adaptive", "abl_param_unconst_odd", 8, False),
    ("H6a_n16", "H6 (fixed-point): Fixed-Point vs Unconstrained Odd Cheby (n=16)", "main_polysort_adaptive", "abl_param_unconst_odd", 16, False),
    ("H6b_n8", "H6 (wiring): Bitonic vs Random Fixed Wiring (n=8)", "main_polysort_adaptive", "abl_wire_random_fixed", 8, False),
    ("H6b_n16", "H6 (wiring): Bitonic vs Random Fixed Wiring (n=16)", "main_polysort_adaptive", "abl_wire_random_fixed", 16, False),
    ("H6c_n8", "H6 (wiring): Batcher Odd-Even Merge vs Bitonic (n=8)", "abl_wire_oddeven_merge", "main_polysort_adaptive", 8, False),
    ("H6c_n16", "H6 (wiring): Batcher Odd-Even Merge vs Bitonic (n=16)", "abl_wire_oddeven_merge", "main_polysort_adaptive", 16, False),
    ("B1_n8", "Baseline: PolySortNet-Adaptive-m2 vs ReLU-MLP (n=8)", "main_polysort_adaptive_m2", "base_relu_mlp", 8, True),
    ("B1_n16", "Baseline: PolySortNet-Adaptive-m2 vs ReLU-MLP (n=16)", "main_polysort_adaptive_m2", "base_relu_mlp", 16, True),
    ("B2_n8", "Baseline: PolySortNet-Adaptive-m2 vs DiffSort-Cauchy (n=8)", "main_polysort_adaptive_m2", "base_diffsort_cauchy", 8, True),
    ("B2_n16", "Baseline: PolySortNet-Adaptive-m2 vs DiffSort-Cauchy (n=16)", "main_polysort_adaptive_m2", "base_diffsort_cauchy", 16, True),
    ("B3_n8", "Baseline: PolySortNet-Adaptive-m2 vs SetTransformer (n=8)", "main_polysort_adaptive_m2", "base_set_transformer", 8, True),
    ("B3_n16", "Baseline: PolySortNet-Adaptive-m2 vs SetTransformer (n=16)", "main_polysort_adaptive_m2", "base_set_transformer", 16, True),
]


def _load_eval(run_id: str) -> Dict[str, Any] | None:
    p = RESULTS_DIR / "runs" / run_id / "eval_report.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _load_per_list(run_id: str, metric: str = "mae") -> np.ndarray | None:
    p = RESULTS_DIR / "runs" / run_id / "test_per_list.npz"
    if not p.exists():
        return None
    return np.load(p)[metric]


def build_all_summaries(num_boot: int = 10_000) -> Dict[str, Any]:
    runs_dir = RESULTS_DIR / "runs"
    all_run_rows: List[Dict[str, Any]] = []
    for eval_path in sorted(runs_dir.glob("*/eval_report.json")):
        d = json.loads(eval_path.read_text(encoding="utf-8"))
        t = d["test"]
        ood = d.get("ood", {})
        all_run_rows.append({
            "run_id": d["run_id"],
            "model_name": d["model_name"],
            "n": d["n"],
            "seed": d["seed"],
            "is_pure_polynomial": d["is_pure_polynomial"],
            "n_params": d["n_params"],
            "seq_mult_depth": d.get("seq_mult_depth"),
            "latency_ms_per_1k": d.get("latency_ms_per_1k"),
            "mae": t["mae"],
            "rmse": t["rmse"],
            "linf_mean": t["linf_mean"],
            "w1": t["w1"],
            "sum_err": t["sum_err"],
            "mono_rate": t["mono_rate"],
            "pair_acc": t["pair_acc"],
            "kendall_tau": t["kendall_tau"],
            "exact_sort_0.05": t["exact_sort_0.05"],
            "exact_sort_0.01": t["exact_sort_0.01"],
            "r_stable": d["r_stable"],
            "ood_gaussian_mae": ood.get("ood_gaussian", {}).get("mae", float("nan")),
            "ood_duplicates_50_mae": ood.get("ood_duplicates_50", {}).get("mae", float("nan")),
            "ood_nearties_1e-3_mae": ood.get("ood_nearties_1e-3", {}).get("mae", float("nan")),
            "ood_shift_100_mae": ood.get("ood_shift_100", {}).get("mae", float("nan")),
            "ood_scale_1.5_mae": ood.get("ood_scale_1.5", {}).get("mae", float("nan")),
            "ood_scale_2_mae": ood.get("ood_scale_2", {}).get("mae", float("nan")),
        })

    # Write flat CSV table
    csv_path = RESULTS_DIR / "summary_table.csv"
    if all_run_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_run_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_run_rows)

    # Multi-seed summary across core models
    multiseed: Dict[str, Any] = {}
    for n in (8, 16):
        multiseed[str(n)] = []
        for prefix, label, group in CORE_PREFIXES:
            seed_reports = []
            per_list_maes = []
            for s in DEFAULT_SEEDS:
                rid = f"{prefix}_n{n}_s{s}"
                rep = _load_eval(rid)
                if rep is not None:
                    seed_reports.append(rep)
                    arr = _load_per_list(rid, "mae")
                    if arr is not None:
                        per_list_maes.append(arr)
            if not seed_reports:
                continue
            maes = [r["test"]["mae"] for r in seed_reports]
            rmses = [r["test"]["rmse"] for r in seed_reports]
            w1s = [r["test"]["w1"] for r in seed_reports]
            monos = [r["test"]["mono_rate"] for r in seed_reports]
            taus = [r["test"]["kendall_tau"] for r in seed_reports]
            ex05s = [r["test"]["exact_sort_0.05"] for r in seed_reports]
            ex01s = [r["test"]["exact_sort_0.01"] for r in seed_reports]
            rstabs = [r["r_stable"] for r in seed_reports]
            # Pooled per-list bootstrap CI across all available seeds
            pooled = np.concatenate(per_list_maes) if per_list_maes else np.asarray(maes)
            b_mean, b_lo, b_hi = bootstrap_ci(pooled, num_resamples=num_boot, seed=2026 + n)
            multiseed[str(n)].append({
                "prefix": prefix,
                "label": label,
                "group": group,
                "n": n,
                "n_seeds": len(seed_reports),
                "seeds": [r["seed"] for r in seed_reports],
                "is_pure_polynomial": seed_reports[0]["is_pure_polynomial"],
                "n_params": seed_reports[0]["n_params"],
                "mae_mean": float(np.mean(maes)),
                "mae_std": float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
                "mae_min": float(np.min(maes)),
                "mae_max": float(np.max(maes)),
                "mae_boot_mean": b_mean,
                "mae_boot_ci95_low": b_lo,
                "mae_boot_ci95_high": b_hi,
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
                "w1_mean": float(np.mean(w1s)),
                "mono_mean": float(np.mean(monos)),
                "mono_std": float(np.std(monos, ddof=1)) if len(monos) > 1 else 0.0,
                "tau_mean": float(np.mean(taus)),
                "tau_std": float(np.std(taus, ddof=1)) if len(taus) > 1 else 0.0,
                "exact05_mean": float(np.mean(ex05s)),
                "exact05_std": float(np.std(ex05s, ddof=1)) if len(ex05s) > 1 else 0.0,
                "exact01_mean": float(np.mean(ex01s)),
                "r_stable_mean": float(np.mean(rstabs)),
                "r_stable_min": float(np.min(rstabs)),
                "ood_shift100_mean": float(np.mean([r["ood"]["ood_shift_100"]["mae"] for r in seed_reports])),
                "ood_nearties1e3_mean": float(np.mean([r["ood"]["ood_nearties_1e-3"]["mae"] for r in seed_reports])),
                "ood_dup50_mean": float(np.mean([r["ood"]["ood_duplicates_50"]["mae"] for r in seed_reports])),
                "ood_gauss_mean": float(np.mean([r["ood"]["ood_gaussian"]["mae"] for r in seed_reports])),
            })

    (RESULTS_DIR / "multiseed_summary.json").write_text(json.dumps(multiseed, indent=2), encoding="utf-8")

    # Paired Wilcoxon signed-rank tests + Holm-Bonferroni correction
    stat_rows: List[Dict[str, Any]] = []
    for tid, desc, pa, pb, n, use_ms in PREREGISTERED_TESTS:
        seeds_to_pool = DEFAULT_SEEDS if use_ms else (42,)
        arrs_a, arrs_b = [], []
        seed_means_a, seed_means_b = [], []
        for s in seeds_to_pool:
            a = _load_per_list(f"{pa}_n{n}_s{s}", "mae")
            b = _load_per_list(f"{pb}_n{n}_s{s}", "mae")
            if a is not None and b is not None:
                arrs_a.append(a)
                arrs_b.append(b)
                seed_means_a.append(float(np.mean(a)))
                seed_means_b.append(float(np.mean(b)))
        if not arrs_a:
            continue
        vec_a = np.concatenate(arrs_a)
        vec_b = np.concatenate(arrs_b)
        res = paired_wilcoxon_test(vec_a, vec_b, num_boot=num_boot, seed=2026)
        mean_a = float(np.mean(vec_a))
        mean_b = float(np.mean(vec_b))
        ratio = mean_b / mean_a if mean_a > 0 else float("nan")
        stat_rows.append({
            "test_id": tid,
            "description": desc,
            "n": n,
            "n_seeds_pooled": len(arrs_a),
            "model_a": pa,
            "model_b": pb,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "ratio_b_over_a": ratio,
            "seed_wins_a_beats_b": sum(1 for x, y in zip(seed_means_a, seed_means_b) if x < y),
            **res,
        })

    # Apply Holm-Bonferroni across entire family of preregistered tests
    adj_ps = holm_bonferroni([r["p_value"] for r in stat_rows])
    for r, adj_p in zip(stat_rows, adj_ps):
        r["p_holm"] = adj_p
        r["significant_0.05"] = bool(adj_p < 0.05)

    (RESULTS_DIR / "statistical_tests.json").write_text(json.dumps(stat_rows, indent=2), encoding="utf-8")
    return {
        "total_runs": len(all_run_rows),
        "multiseed": multiseed,
        "statistical_tests": stat_rows,
    }


if __name__ == "__main__":
    out = build_all_summaries()
    print(f"Aggregated {out['total_runs']} runs and {len(out['statistical_tests'])} Holm-Bonferroni paired Wilcoxon tests.")
    for r in out["statistical_tests"]:
        sig = "*" if r["significant_0.05"] else "ns"
        print(
            f"{r['test_id']:9s} (seeds={r['n_seeds_pooled']}) | A={r['mean_a']:.5f} vs B={r['mean_b']:.5f} "
            f"| diff={r['mean_diff']:+.5f} [{r['ci_low']:+.5f}, {r['ci_high']:+.5f}] "
            f"| r_rb={r['rank_biserial']:+.3f} | p_holm={r['p_holm']:.2e} ({sig})"
        )
