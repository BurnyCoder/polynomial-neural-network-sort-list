"""Comprehensive evaluation harness for value fidelity, ordering accuracy, OOD, and zero-shot scaling."""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from pnn_sort.config import RESULTS_DIR, SCALE_SWEEP_FACTORS, ZERO_SHOT_LENGTHS
from pnn_sort.data import generate_ood_suites, generate_split
from pnn_sort.models import count_parameters


def _batched_forward(model: nn.Module, x: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            outs.append(model(x[i : i + batch_size]))
    return torch.cat(outs, dim=0)


def compute_per_list_metrics(pred: torch.Tensor, target: torch.Tensor, x_in: torch.Tensor) -> Dict[str, np.ndarray]:
    """Compute exact per-list metrics as float64 numpy arrays for bootstrap CIs and Wilcoxon tests."""
    p = pred.detach().double().cpu().numpy()
    y = target.detach().double().cpu().numpy()
    x = x_in.detach().double().cpu().numpy()
    n_lists, n = p.shape

    finite_mask = np.isfinite(p).all(axis=-1)
    if not finite_mask.all():
        # Clamp non-finite outputs to nan so summaries report divergence honestly
        p = np.where(np.isfinite(p), p, np.nan)

    abs_err = np.abs(p - y)
    mae = np.mean(abs_err, axis=-1)
    mse = np.mean((p - y) ** 2, axis=-1)
    linf = np.max(abs_err, axis=-1)

    # Multiset Wasserstein-1 distance: MAE after sorting predictions (isolates value distortion from order)
    p_sorted = np.sort(p, axis=-1)
    w1 = np.mean(np.abs(p_sorted - y), axis=-1)

    # Sum conservation error
    sum_err = np.abs(np.sum(p, axis=-1) - np.sum(x, axis=-1))

    # Ordering metrics
    mono_tol = np.all(p[:, :-1] <= p[:, 1:] + 1e-7, axis=-1).astype(np.float64)
    mono_exact = np.all(p[:, :-1] <= p[:, 1:], axis=-1).astype(np.float64)

    # Pairwise order accuracy & Kendall's tau against ascending order (1..n)
    # Pair (i < j) is concordant iff p_i <= p_j + 1e-7
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_ok = (p[:, i_idx] <= p[:, j_idx] + 1e-7).astype(np.float64)
    pair_acc = np.mean(pair_ok, axis=-1)
    kendall_tau = 2.0 * pair_acc - 1.0

    # Value-matched exact sort rates (prevents vacuous constant-output monotonicity gaming)
    exact_005 = (mono_tol * (linf < 0.05)).astype(np.float64)
    exact_001 = (mono_tol * (linf < 0.01)).astype(np.float64)
    exact_0001 = (mono_tol * (linf < 0.001)).astype(np.float64)

    return {
        "mae": mae,
        "mse": mse,
        "linf": linf,
        "w1": w1,
        "sum_err": sum_err,
        "mono_rate": mono_tol,
        "mono_exact_rate": mono_exact,
        "pair_acc": pair_acc,
        "kendall_tau": kendall_tau,
        "exact_sort_0.05": exact_005,
        "exact_sort_0.01": exact_001,
        "exact_sort_0.001": exact_0001,
        "finite_rate": finite_mask.astype(np.float64),
    }


def summarize_per_list(per_list: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Aggregate per-list metric arrays into scalar summary stats (handling NaNs/overflows honestly)."""
    finite_rate = float(np.mean(per_list["finite_rate"]))
    if finite_rate < 1.0:
        # Report finite subset stats only if any finite exist, else inf
        mask = per_list["finite_rate"] > 0.5
        if not np.any(mask):
            return {
                "finite_rate": 0.0,
                "mae": float("inf"),
                "rmse": float("inf"),
                "linf_mean": float("inf"),
                "linf_p99": float("inf"),
                "w1": float("inf"),
                "sum_err": float("inf"),
                "mono_rate": 0.0,
                "mono_exact_rate": 0.0,
                "pair_acc": 0.0,
                "kendall_tau": -1.0,
                "exact_sort_0.05": 0.0,
                "exact_sort_0.01": 0.0,
                "exact_sort_0.001": 0.0,
            }
        # If partially diverged, mark mae/rmse as inf so divergence is never hidden
        return {
            "finite_rate": finite_rate,
            "mae": float("inf"),
            "rmse": float("inf"),
            "linf_mean": float("inf"),
            "linf_p99": float("inf"),
            "w1": float("inf"),
            "sum_err": float("inf"),
            "mono_rate": float(np.mean(np.where(mask, per_list["mono_rate"], 0.0))),
            "mono_exact_rate": float(np.mean(np.where(mask, per_list["mono_exact_rate"], 0.0))),
            "pair_acc": float(np.mean(np.where(mask, per_list["pair_acc"], 0.0))),
            "kendall_tau": float(np.mean(np.where(mask, per_list["kendall_tau"], -1.0))),
            "exact_sort_0.05": float(np.mean(np.where(mask, per_list["exact_sort_0.05"], 0.0))),
            "exact_sort_0.01": float(np.mean(np.where(mask, per_list["exact_sort_0.01"], 0.0))),
            "exact_sort_0.001": float(np.mean(np.where(mask, per_list["exact_sort_0.001"], 0.0))),
        }

    return {
        "finite_rate": 1.0,
        "mae": float(np.mean(per_list["mae"])),
        "rmse": float(math.sqrt(float(np.mean(per_list["mse"])))),
        "linf_mean": float(np.mean(per_list["linf"])),
        "linf_p99": float(np.percentile(per_list["linf"], 99)),
        "w1": float(np.mean(per_list["w1"])),
        "sum_err": float(np.mean(per_list["sum_err"])),
        "mono_rate": float(np.mean(per_list["mono_rate"])),
        "mono_exact_rate": float(np.mean(per_list["mono_exact_rate"])),
        "pair_acc": float(np.mean(per_list["pair_acc"])),
        "kendall_tau": float(np.mean(per_list["kendall_tau"])),
        "exact_sort_0.05": float(np.mean(per_list["exact_sort_0.05"])),
        "exact_sort_0.01": float(np.mean(per_list["exact_sort_0.01"])),
        "exact_sort_0.001": float(np.mean(per_list["exact_sort_0.001"])),
    }


def measure_latency_ms(model: nn.Module, n: int, device: torch.device, dtype: torch.dtype, num_lists: int = 1000) -> float:
    """Measure average inference wall-clock time in milliseconds per 1,000 lists."""
    model.eval()
    x = torch.rand(num_lists, n, device=device, dtype=dtype) * 2.0 - 1.0
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        iters = 30
        for _ in range(iters):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / iters
    return float(elapsed * 1000.0)


def evaluate_model_full(
    model: nn.Module,
    run_id: str,
    model_name: str,
    n: int,
    test_size: int = 10_000,
    ood_size: int = 2_000,
    seed: int = 42,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
    eval_zero_shot: bool = True,
    eval_float64: bool = True,
    results_root: Path = RESULTS_DIR,
) -> Dict[str, Any]:
    """Run complete held-out test, OOD stress suite, float64 check, and zero-shot length evaluation."""
    dev = torch.device(device)
    model = model.to(device=dev, dtype=dtype)
    model.eval()

    run_dir = results_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Held-out in-distribution test set (same seed=9999 test stream across ALL models for paired Wilcoxon!)
    x_test, y_test = generate_split("test", n, test_size, seed=9999, dtype=dtype)
    x_test_dev = x_test.to(dev)
    pred_test = _batched_forward(model, x_test_dev)
    per_list_test = compute_per_list_metrics(pred_test, y_test, x_test)
    test_summary = summarize_per_list(per_list_test)

    # Save exact per-list arrays for paired bootstrap & Wilcoxon tests
    np.savez_compressed(
        run_dir / "test_per_list.npz",
        mae=per_list_test["mae"],
        linf=per_list_test["linf"],
        mono_rate=per_list_test["mono_rate"],
        kendall_tau=per_list_test["kendall_tau"],
        w1=per_list_test["w1"],
        sum_err=per_list_test["sum_err"],
    )

    # Save 32 concrete output examples (first 16 ID + 16 OOD cases) for visualizations
    sample_examples: Dict[str, Any] = {
        "in_dist": {
            "x": x_test[:16].cpu().numpy().tolist(),
            "y_true": y_test[:16].cpu().numpy().tolist(),
            "y_pred": pred_test[:16].detach().cpu().numpy().tolist(),
        }
    }

    # 2. All OOD stress suites
    ood_suites = generate_ood_suites(n, size=ood_size, seed=9999, dtype=dtype)
    ood_results: Dict[str, Dict[str, float]] = {}
    for ood_name, (xo, yo) in ood_suites.items():
        po = _batched_forward(model, xo.to(dev))
        pl = compute_per_list_metrics(po, yo, xo)
        ood_results[ood_name] = summarize_per_list(pl)
        if ood_name in ("ood_nearties_1e-3", "ood_duplicates_50", "ood_shift_100", "ood_scale_2", "ood_scale_5"):
            sample_examples[ood_name] = {
                "x": xo[:8].cpu().numpy().tolist(),
                "y_true": yo[:8].cpu().numpy().tolist(),
                "y_pred": po[:8].detach().cpu().numpy().tolist(),
            }

    # Empirical Scale Stability Radius R_stable: largest s with relative MAE (MAE / s) < 0.05
    r_stable = 0.0
    for s in SCALE_SWEEP_FACTORS:
        mae_s = ood_results[f"ood_scale_{s:g}"]["mae"]
        if math.isfinite(mae_s) and (mae_s / s) < 0.05:
            r_stable = float(s)
        else:
            break

    # 3. Float64 verification on test & shift_100 (disentangles true math error from float32 roundoff)
    f64_results: Dict[str, Dict[str, float]] = {}
    if eval_float64:
        model_f64 = copy.deepcopy(model).double().to(dev)
        x_t64, y_t64 = generate_split("test", n, min(test_size, 2000), seed=9999, dtype=torch.float64)
        p_t64 = _batched_forward(model_f64, x_t64.to(dev))
        f64_results["test_f64"] = summarize_per_list(compute_per_list_metrics(p_t64, y_t64, x_t64))

        x_sh64 = x_t64[:1000] + 100.0
        y_sh64 = y_t64[:1000] + 100.0
        p_sh64 = _batched_forward(model_f64, x_sh64.to(dev))
        shift_equiv_err = float(
            torch.max(torch.abs((p_sh64.cpu() - 100.0) - p_t64[:1000].cpu())).item()
        )
        f64_results["ood_shift_100_f64"] = summarize_per_list(compute_per_list_metrics(p_sh64, y_sh64, x_sh64))
        f64_results["ood_shift_100_f64"]["shift_equivariance_linf"] = shift_equiv_err

    # 4. Zero-shot list-length generalization (for variable-length models)
    zero_shot_results: Dict[str, Dict[str, float]] = {}
    if eval_zero_shot and getattr(model, "supports_variable_length", False):
        for new_n in ZERO_SHOT_LENGTHS:
            zs_model = model.for_length(new_n).to(dev)
            xz, yz = generate_split("test_zeroshot", new_n, 2000, seed=9999, dtype=dtype)
            pz = _batched_forward(zs_model, xz.to(dev), batch_size=512 if new_n >= 64 else 2048)
            zero_shot_results[f"n={new_n}"] = summarize_per_list(compute_per_list_metrics(pz, yz, xz))

    # 5. Latency
    lat_ms = measure_latency_ms(model, n=n, device=dev, dtype=dtype)

    report: Dict[str, Any] = {
        "run_id": run_id,
        "model_name": model_name,
        "n": n,
        "seed": seed,
        "is_pure_polynomial": bool(getattr(model, "is_pure_polynomial", False)),
        "supports_variable_length": bool(getattr(model, "supports_variable_length", False)),
        "n_params": count_parameters(model),
        "effective_degree": getattr(model, "effective_degree", None),
        "seq_mult_depth": getattr(model, "seq_mult_depth", None),
        "latency_ms_per_1k": lat_ms,
        "r_stable": r_stable,
        "test": test_summary,
        "ood": ood_results,
        "float64_check": f64_results,
        "zero_shot_length": zero_shot_results,
    }

    (run_dir / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (run_dir / "examples.json").write_text(json.dumps(sample_examples), encoding="utf-8")
    return report
