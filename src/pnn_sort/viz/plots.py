"""Publication-grade matplotlib visualizations (zero graph crimes, colorblind-safe Okabe-Ito palette)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pnn_sort.models import PolySortNet, build_analytical_cheon_sort, build_analytical_remez_sort

# Okabe-Ito colorblind-safe palette
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#333333",  # dark gray
]


def _style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.7)


def plot_architecture_comparison(
    rows: List[Dict[str, Any]],
    title: str,
    out_path: Path,
    metric: str = "mae",
    log_scale: bool = True,
) -> None:
    """Horizontal bar chart comparing architectures with optional 95% CI / std error bars."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Sort best-to-worst (lowest MAE at top, highest mono/tau at top)
    reverse = metric not in ("mae", "rmse", "linf_mean", "w1")
    clean = [r for r in rows if np.isfinite(r.get(metric, np.nan))]
    clean.sort(key=lambda r: r[metric], reverse=reverse)
    if not clean:
        return

    labels = [r["label"] for r in clean]
    vals = [r[metric] for r in clean]
    errs = [r.get(f"{metric}_err", 0.0) for r in clean]
    colors = [
        PALETTE[0] if r.get("group") == "main_pnn"
        else PALETTE[2] if r.get("group") == "analytical"
        else PALETTE[1] if r.get("group") == "non_poly"
        else PALETTE[7]
        for r in clean
    ]

    fig, ax = plt.subplots(figsize=(9.5, max(4.2, 0.42 * len(clean) + 1.4)))
    y_pos = np.arange(len(clean))
    ax.barh(y_pos, vals, xerr=errs if any(e > 0 for e in errs) else None, color=colors, alpha=0.9, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    if log_scale and all(v > 0 for v in vals):
        ax.set_xscale("log")
    else:
        ax.set_xlim(left=0.0)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    xlabel_map = {
        "mae": "Held-Out Value MAE (lower is better)",
        "mono_rate": "Full-List Monotonicity Rate (higher is better)",
        "pair_acc": "Pairwise Order Accuracy (higher is better)",
        "kendall_tau": "Kendall's Tau (higher is better)",
    }
    ax.set_xlabel(xlabel_map.get(metric, metric), fontsize=10.5)
    _style_ax(ax)

    # Direct value labels for readability
    for i, v in enumerate(vals):
        txt = f"{v:.5f}" if metric in ("mae", "rmse", "w1") else f"{v*100:.2f}%"
        ax.text(v * (1.08 if log_scale else 1.0) + (0.0 if log_scale else 0.005), i, txt, va="center", fontsize=8.5)

    # Legend for model categories
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=PALETTE[0], label="Trained Pure PNN (Ours)"),
        Patch(facecolor=PALETTE[2], label="Analytical CKKS PNN (Zero-Train)"),
        Patch(facecolor=PALETTE[1], label="Non-Polynomial Baseline"),
        Patch(facecolor=PALETTE[7], label="Control / Lower Bound"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_comparator_transfer_curves(
    models: Dict[str, PolySortNet],
    out_path: Path,
) -> None:
    """Two-panel mechanistic figure: (Left) s(u) vs ideal sign(u); (Right) CAS value error u*|1 - s(u)|."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    u_np = np.linspace(0.0, 1.0, 1000)
    u_t = torch.from_numpy(u_np).double()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax1.plot(u_np, np.ones_like(u_np), "k--", linewidth=1.2, alpha=0.6, label="Ideal sign(u) = 1")

    for idx, (label, model) in enumerate(models.items()):
        color = PALETTE[idx % len(PALETTE)]
        sign_mod = model.sign_shared if model.sign_shared is not None else model.sign_stages[-1]
        sign_d = sign_mod.cpu().double()
        with torch.no_grad():
            s_val = sign_d(u_t).numpy()
        val_err = u_np * np.abs(1.0 - s_val)
        ax1.plot(u_np, s_val, color=color, linewidth=2.0, label=label)
        ax2.plot(u_np, val_err, color=color, linewidth=2.0, label=f"{label} (max={val_err.max():.4f})")

    ax1.set_title("Learned vs. Analytical Sign Approximator s(u)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Normalized Half-Difference Δ = (b - a)/2")
    ax1.set_ylabel("Comparator Response s(Δ)")
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(-0.05, 1.15)
    ax1.legend(fontsize=8.5, loc="lower right")
    _style_ax(ax1)

    ax2.set_title("Single-Gate Value Error e_val(Δ) = Δ · |1 - s(Δ)|", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Normalized Half-Difference Δ = (b - a)/2")
    ax2.set_ylabel("Absolute Value Error |min_s(a,b) - min(a,b)|")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(bottom=0.0)
    ax2.legend(fontsize=8.5, loc="upper right")
    _style_ax(ax2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_scale_cliff(
    scale_curves: Dict[str, Dict[float, float]],
    out_path: Path,
) -> None:
    """Plot empirical relative MAE (MAE / s) across scale factors s in [0.1, 5.0] showing polynomial cliff."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for idx, (label, curve) in enumerate(scale_curves.items()):
        xs = sorted(curve.keys())
        ys = [min(curve[s] / s, 1e3) if np.isfinite(curve[s]) else 1e3 for s in xs]
        ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=5, color=PALETTE[idx % len(PALETTE)], label=label)

    ax.axvline(1.0, color="#555555", linestyle=":", linewidth=1.4, label="Training Domain Boundary (s = 1.0)")
    ax.axhline(0.05, color="#cc0000", linestyle="--", linewidth=1.1, alpha=0.7, label="Stability Threshold (rel MAE = 0.05)")
    ax.set_yscale("log")
    ax.set_xlabel("Input Scale Multiplier s  (x_test = s · Uniform[-1, 1]^n)", fontsize=10.5)
    ax.set_ylabel("Relative Value Error (MAE / s, log scale, capped at 10³)", fontsize=10.5)
    ax.set_title("Pure Polynomial Scale-Extrapolation Cliff vs. Non-Poly Baselines", fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left")
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_zero_shot_length(
    length_curves: Dict[str, Dict[int, float]],
    out_path: Path,
) -> None:
    """Plot zero-shot Value MAE across list lengths n = 4..128 for models trained at n = 8."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    for idx, (label, curve) in enumerate(length_curves.items()):
        xs = sorted(curve.keys())
        ys = [curve[k] for k in xs]
        ax.plot(xs, ys, marker="s", linewidth=2.2, markersize=5.5, color=PALETTE[idx % len(PALETTE)], label=label)

    ax.axvline(8, color="#555555", linestyle=":", linewidth=1.4, label="Training Length (n = 8)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([4, 8, 16, 32, 64, 128])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Evaluation List Length n (Zero-Shot Extrapolation up to 16× Training Length)", fontsize=10.5)
    ax.set_ylabel("Held-Out Value MAE (log scale)", fontsize=10.5)
    ax.set_title("Zero-Shot List-Length Generalization (n = 4 to n = 128)", fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left")
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_output_examples(
    examples_by_model: Dict[str, Dict[str, List[float]]],
    title: str,
    out_path: Path,
) -> None:
    """Visual comparison of unsorted input, ground-truth sort, and predicted sorted values across models."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(examples_by_model.keys())
    n_models = len(names)
    cols = min(3, n_models)
    rows_n = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows_n, cols, figsize=(4.5 * cols, 3.3 * rows_n), squeeze=False)
    for idx, name in enumerate(names):
        ax = axes[idx // cols][idx % cols]
        ex = examples_by_model[name]
        y_true = np.asarray(ex["y_true"], dtype=np.float64)
        y_pred = np.asarray(ex["y_pred"], dtype=np.float64)
        x_in = np.asarray(ex["x"], dtype=np.float64)
        pos = np.arange(1, len(y_true) + 1)

        ax.scatter(pos, x_in, color="#bbbbbb", marker="x", s=34, label="Unsorted Input x", zorder=2)
        ax.plot(pos, y_true, "k-o", linewidth=1.8, markersize=4.5, label="Ground Truth sort(x)", zorder=3)
        y_pred_clip = np.clip(y_pred, -3.0, 3.0)
        mae = float(np.mean(np.abs(y_pred - y_true))) if np.isfinite(y_pred).all() else float("inf")
        mono = bool(np.all(y_pred[:-1] <= y_pred[1:] + 1e-7)) if np.isfinite(y_pred).all() else False
        ax.plot(pos, y_pred_clip, "--s", color=PALETTE[idx % len(PALETTE)], linewidth=1.8, markersize=4.5, label="Model Output", zorder=4)
        mae_str = f"{mae:.4f}" if np.isfinite(mae) and mae < 100 else (f"{mae:.1e}" if np.isfinite(mae) else "INF")
        ax.set_title(f"{name}\nMAE={mae_str} | Mono={'Yes' if mono else 'No'}", fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Output Rank Position k", fontsize=8.5)
        ax.set_ylabel("Value", fontsize=8.5)
        _style_ax(ax)
        if idx == 0:
            ax.legend(fontsize=7.5, loc="upper left")

    for j in range(n_models, rows_n * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    log_paths: Dict[str, Path],
    title: str,
    out_path: Path,
) -> None:
    """Plot validation MAE and pairwise order accuracy learning curves across training epochs."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for idx, (label, path) in enumerate(log_paths.items()):
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            continue
        epochs = [r["epoch"] for r in rows]
        maes = [r["val_mae"] for r in rows]
        accs = [r["val_pair_acc"] * 100.0 for r in rows]
        c = PALETTE[idx % len(PALETTE)]
        ax1.plot(epochs, maes, linewidth=2.0, color=c, label=label)
        ax2.plot(epochs, accs, linewidth=2.0, color=c, label=label)

    ax1.set_yscale("log")
    ax1.set_title("Validation Value MAE During Training", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation MAE (log scale)")
    ax1.legend(fontsize=8.2)
    _style_ax(ax1)

    ax2.set_title("Validation Pairwise Order Accuracy During Training", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Pairwise Order Accuracy (%)")
    ax2.set_ylim(50.0, 100.5)
    ax2.legend(fontsize=8.2, loc="lower right")
    _style_ax(ax2)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
