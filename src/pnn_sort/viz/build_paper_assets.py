"""Generate all publication figures (paper + README + docs) and GitHub Pages explorer payloads."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from pnn_sort.config import CHECKPOINTS_DIR, DOCS_DIR, PROJECT_ROOT, REPORTS_DIR, RESULTS_DIR
from pnn_sort.models import build_model
from pnn_sort.publish import export_browser_poly_weights
from pnn_sort.viz.plots import PALETTE, _style_ax


PAPER_FIG_DIR = REPORTS_DIR / "paper" / "figures"
DOCS_FIG_DIR = DOCS_DIR / "figures"


def _save_both(fig: plt.Figure, filename: str) -> None:
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    p1 = PAPER_FIG_DIR / filename
    p2 = DOCS_FIG_DIR / filename
    fig.savefig(p1, dpi=260, bbox_inches="tight", facecolor="white")
    shutil.copy2(p1, p2)
    plt.close(fig)


def _load_eval(run_id: str) -> Dict[str, Any]:
    return json.loads((RESULTS_DIR / "runs" / run_id / "eval_report.json").read_text(encoding="utf-8"))


def _load_trained_model(run_id: str, model_name: str, n: int, **kwargs: Any) -> torch.nn.Module:
    m = build_model(model_name, n=n, **kwargs)
    ckpt = CHECKPOINTS_DIR / run_id / "best.pt"
    if ckpt.exists():
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        m.load_state_dict(sd["model"])
    return m.double().eval()


def build_fig1_architecture() -> None:
    """Publication architecture diagram: Bitonic CAS wiring + Pure Polynomial Gate + PNN Inductive-Bias Ladder."""
    fig, ax = plt.subplots(figsize=(13.0, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#F5F8FC", ec="#0072B2", lw=1.6, fs=9.2, bold=False, tc="#111111"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.14",
            facecolor=fc, edgecolor=ec, linewidth=lw
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", color=tc)

    ax.text(6.5, 5.38, "PolySortNet Architecture & Pure-Polynomial Inductive-Bias Ladder",
            ha="center", va="center", fontsize=13.5, fontweight="bold", color="#111111")

    # Panel A: Bitonic Sorting Network (n=4 illustration)
    ax.text(2.15, 4.92, "(A) Fixed Sorting-Network Wiring (Bitonic / Odd-Even)",
            ha="center", va="center", fontsize=10.2, fontweight="bold", color="#0072B2")
    wire_ys = [4.35, 3.85, 3.35, 2.85]
    labels_in = ["x₁", "x₂", "x₃", "x₄"]
    labels_out = ["ŷ₁ (min)", "ŷ₂", "ŷ₃", "ŷ₄ (max)"]
    for i, y in enumerate(wire_ys):
        ax.plot([0.55, 3.65], [y, y], color="#444444", lw=1.6, zorder=1)
        ax.text(0.32, y, labels_in[i], ha="right", va="center", fontsize=9.5, fontweight="bold")
        ax.text(3.75, y, labels_out[i], ha="left", va="center", fontsize=8.8)

    stages = [
        (1.15, [(0, 1), (2, 3)], "Stage 0"),
        (2.10, [(0, 3), (1, 2)], "Stage 1"),
        (3.05, [(0, 1), (2, 3)], "Stage 2"),
    ]
    for sx, pairs, sname in stages:
        ax.text(sx, 2.52, sname, ha="center", va="center", fontsize=8.3, color="#555555")
        for idx_p, (a, b) in enumerate(pairs):
            ox = sx + (idx_p - 0.5) * 0.16 if len(pairs) > 1 and abs(a - b) > 1 else sx
            ax.plot([ox, ox], [wire_ys[a], wire_ys[b]], color="#0072B2", lw=2.3, zorder=3)
            ax.scatter([ox, ox], [wire_ys[a], wire_ys[b]], s=48, color="#0072B2", zorder=4)

    ax.text(2.15, 2.18, "Zero-shot extensible to any power-of-2 length n\nExact sum conservation & shift equivariance",
            ha="center", va="center", fontsize=8.3, color="#333333",
            bbox=dict(boxstyle="round,pad=0.25", fc="#EEF5FB", ec="#0072B2", lw=0.9))

    # Arrow A -> B
    ax.annotate("", xy=(4.65, 3.60), xytext=(4.25, 3.60),
                arrowprops=dict(arrowstyle="->", lw=2.0, color="#0072B2"))

    # Panel B: Pure Polynomial CAS Gate & Composite Odd Chebyshev Block
    ax.text(8.75, 4.92, "(B) Strictly Pure-Polynomial Compare-and-Swap (CAS) Gate",
            ha="center", va="center", fontsize=10.2, fontweight="bold", color="#D55E00")
    box(4.65, 3.70, 3.65, 0.95,
        "Pairwise Inputs $(a, b)$\n$\\mu = (a + b)/2, \\quad \\Delta = (b - a)/2$\nPure linear maps (shift-invariant $\\Delta$)",
        fc="#FFF7F0", ec="#D55E00", fs=8.8)
    box(4.65, 2.35, 3.65, 1.00,
        "Pure-Poly Output Swap\n$\\widehat{\\min}(a, b) = \\mu - \\Delta \\cdot s_k(\\Delta)$\n$\\widehat{\\max}(a, b) = \\mu + \\Delta \\cdot s_k(\\Delta)$\n(Only $+, -, \\times$ — FX Verified!)",
        fc="#FFF7F0", ec="#D55E00", fs=8.8, bold=False)
    ax.annotate("", xy=(6.47, 3.40), xytext=(6.47, 3.68), arrowprops=dict(arrowstyle="->", lw=1.6, color="#D55E00"))

    box(8.62, 2.30, 4.18, 2.38,
        "$L$-Layer Composite Odd Chebyshev $s_k(\\Delta)$\n"
        "$u_0 = \\Delta, \\; u_\\ell = h_{k,\\ell}(u_{\\ell-1}), \\; s_k(\\Delta) = u_L$\n\n"
        "Fixed-Point Layer ($m$ terms):\n"
        "$h(u) = u + u(1 - u^2)\\sum_{j=0}^{m-1} c_j T_{2j}(u)$\n\n"
        "• Guarantees $h(0)=0, h(\\pm 1)=\\pm 1$, odd symmetry\n"
        "• Degree $= (2m+1)^L$  (e.g., $3^3=27, 3^4=81, 5^3=125$)\n"
        "• Parameters: only 3–60 scalars total!",
        fc="#F2FBF7", ec="#009E73", fs=8.3)
    ax.annotate("", xy=(8.58, 3.95), xytext=(8.33, 3.95), arrowprops=dict(arrowstyle="->", lw=1.6, color="#009E73"))
    ax.annotate("", xy=(8.33, 2.85), xytext=(8.58, 2.85), arrowprops=dict(arrowstyle="->", lw=1.6, color="#009E73"))

    # Panel C: Pure-PNN Inductive-Bias Ladder across bottom
    ax.text(6.5, 1.65, "(C) Five-Tier Pure-Polynomial Inductive-Bias Ladder Evaluated in This Study",
            ha="center", va="center", fontsize=10.2, fontweight="bold", color="#333333")
    tiers = [
        ("1. PolyMLP\n(37–39k params)\nQuadratic $x^2+x$ MLP\nNo symmetry", "#F7F7F7", "#666666"),
        ("2. $\\Pi$-Net NCP-Skip\n(201–206k params)\nHadamard factored\nHigh-order PNN", "#F7F7F7", "#666666"),
        ("3. DeepSets-Poly\n(131k params)\nChebyshev power\nsums ($S_n$ invariant)", "#FDF4E7", "#E69F00"),
        ("4. PolyPairwiseRank\n(12 params)\nConstant-depth $O(n^2)$\nMazzone tie-corr.", "#F8EFF7", "#CC79A7"),
        ("5. PolySortNet (Ours)\n(3–60 params)\nComposite CAS wire\nZero-shot $n=128$", "#EAF4FC", "#0072B2"),
    ]
    w_t = 2.25
    gap = 0.28
    x0 = 0.30
    for idx, (txt, fc, ec) in enumerate(tiers):
        xt = x0 + idx * (w_t + gap)
        box(xt, 0.25, w_t, 1.12, txt, fc=fc, ec=ec, lw=2.0 if idx == 4 else 1.3, fs=8.2, bold=(idx == 4))
        if idx < len(tiers) - 1:
            ax.annotate("", xy=(xt + w_t + gap - 0.03, 0.81), xytext=(xt + w_t + 0.03, 0.81),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color="#555555"))

    _save_both(fig, "fig1_architecture.png")


def build_fig2_main_comparison() -> None:
    """Two-panel bar chart comparing Value MAE (log scale) across n=8 and n=16 with 5-seed std error bars."""
    ms_path = RESULTS_DIR / "multiseed_summary.json"
    ms = json.loads(ms_path.read_text(encoding="utf-8")) if ms_path.exists() else {}

    extra_single = [
        ("main_polysort_shared_m2", "PolySortNet-Shared m=2 (6p)", "main_pnn"),
        ("abl_depth_L4_d3", "PolySortNet-Adaptive L=4 (24/40p)", "main_pnn"),
        ("main_polysort_learnedwire", "PolySortNet-LearnedWire", "main_pnn"),
        ("ana_cheon_fg", "Analytical Cheon f3∘g3 (0p)", "analytical"),
        ("ana_remez_L3", "Analytical Remez L=3 (0p)", "analytical"),
        ("ana_cheon_f1_L3", "Analytical Cheon f1^(3) (0p)", "analytical"),
        ("base_diffsort_logistic", "DiffSort-Logistic (non-poly)", "non_poly"),
        ("ctrl_linear", "Linear Control (0p)", "control"),
        ("ctrl_shuffled", "Shuffled-Target Leakage Ctrl", "control"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.8))
    for col, n in enumerate((8, 16)):
        ax = axes[col]
        rows: List[Dict[str, Any]] = []
        for item in ms.get(str(n), []):
            p_str = f"{item['n_params']}p" if item["n_params"] < 1000 else f"{item['n_params']/1000:.0f}kp"
            rows.append({
                "label": f"{item['label']} ({p_str})",
                "mae": item["mae_mean"],
                "err": item["mae_std"],
                "group": item["group"],
                "ex05": item["exact05_mean"] * 100.0,
            })
        for pref, lbl, grp in extra_single:
            rid = f"{pref}_n{n}_s42"
            p = RESULTS_DIR / "runs" / rid / "eval_report.json"
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                rows.append({
                    "label": lbl,
                    "mae": d["test"]["mae"],
                    "err": 0.0,
                    "group": grp,
                    "ex05": d["test"]["exact_sort_0.05"] * 100.0,
                })
        rows = [r for r in rows if np.isfinite(r["mae"]) and r["mae"] > 0]
        rows.sort(key=lambda r: r["mae"])

        y = np.arange(len(rows))
        vals = [r["mae"] for r in rows]
        errs = [r["err"] for r in rows]
        colors = [
            PALETTE[0] if r["group"] == "main_pnn"
            else PALETTE[2] if r["group"] == "analytical"
            else PALETTE[1] if r["group"] == "non_poly"
            else PALETTE[7]
            for r in rows
        ]
        ax.barh(y, vals, xerr=errs, color=colors, alpha=0.90, capsize=2.8, ecolor="#222222")
        ax.set_yticks(y)
        ax.set_yticklabels([r["label"] for r in rows], fontsize=8.7)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(2e-4, 1.8)
        ax.set_xlabel("Held-Out Test Value MAE (log scale, lower is better)", fontsize=9.8)
        ax.set_title(f"List Length n = {n} (5-Seed Mean ± SD where replicated)", fontsize=11.0, fontweight="bold")
        _style_ax(ax)

        for i, r in enumerate(rows):
            ax.text(r["mae"] * 1.12, i, f"{r['mae']:.4f}  (Ex05: {r['ex05']:.0f}%)",
                    va="center", fontsize=7.6, color="#222222")

    handles = [
        mpatches.Patch(facecolor=PALETTE[0], label="Trained Strictly Pure PNN (Ours / Ladder)"),
        mpatches.Patch(facecolor=PALETTE[2], label="Analytical CKKS Polynomial (Zero-Train)"),
        mpatches.Patch(facecolor=PALETTE[1], label="Non-Polynomial Baseline (Upper Bound)"),
        mpatches.Patch(facecolor=PALETTE[7], label="Trivial / Leakage Control"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True, fontsize=9.0, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Held-Out Sorting Accuracy Across Pure Polynomials, Analytical CKKS, Baselines & Controls",
                 fontsize=13.0, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_both(fig, "fig2_main_comparison.png")


def build_fig3_mechanistic_comparators() -> None:
    """Three-panel mechanistic comparator study: s(u), value error u|1-s(u)|, and stage-adaptive slope profile."""
    u_np = np.linspace(0.0, 1.25, 1000)
    u_t = torch.from_numpy(u_np).double()

    models = {
        "Cheon f1^(3) (deg 27)": _load_trained_model("ana_cheon_f1_L3_n8_s42", "analytical_cheon_f1", 8, num_layers=3),
        "Remez L=3 (deg 27)": _load_trained_model("ana_remez_L3_n8_s42", "analytical_remez", 8, num_layers=3, num_terms=1, delta=0.10),
        "Cheon f3∘g3 (deg 49)": _load_trained_model("ana_cheon_fg_n8_s42", "analytical_cheon_fg", 8),
        "PolySort-Shared m=1 (deg 27)": _load_trained_model("main_polysort_shared_n8_s42", "polysort_shared", 8),
        "PolySort-Shared m=2 (deg 125)": _load_trained_model("main_polysort_shared_m2_n8_s42", "polysort_shared_m2", 8),
        "PolySort-Adapt L=4 (deg 81, St5)": _load_trained_model(
            "abl_depth_L4_d3_n8_s42", "polysort_custom", 8,
            shared=False, num_layers=4, num_terms=1, mode="fixed_point_cheby", init="cheon"
        ),
    }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.4))
    ax1.plot(u_np, np.ones_like(u_np), "k--", lw=1.2, alpha=0.6, label="Ideal sign(u)")
    ax1.axvline(1.0, color="#888888", ls=":", lw=1.1)

    for idx, (lbl, m) in enumerate(models.items()):
        c = PALETTE[idx % len(PALETTE)]
        sign_mod = m.sign_shared if getattr(m, "sign_shared", None) is not None else m.sign_stages[-1]
        with torch.no_grad():
            s_val = sign_mod(u_t).numpy()
        ax1.plot(u_np, s_val, color=c, lw=2.0, label=lbl)
        mask = u_np <= 1.0
        val_err = u_np[mask] * np.abs(1.0 - s_val[mask])
        ax2.plot(u_np[mask], val_err, color=c, lw=2.0, label=f"{lbl} (peak {val_err.max():.3f})")

    ax1.set_title("(A) Polynomial Comparator s(Δ)", fontsize=10.8, fontweight="bold")
    ax1.set_xlabel("Half-Difference Δ = (b − a) / 2")
    ax1.set_ylabel("Comparator Output s(Δ)")
    ax1.set_xlim(0.0, 1.25)
    ax1.set_ylim(-0.1, 1.35)
    ax1.legend(fontsize=7.3, loc="lower right")
    _style_ax(ax1)

    ax2.set_title("(B) Single-Gate Value Error Δ·|1 − s(Δ)|", fontsize=10.8, fontweight="bold")
    ax2.set_xlabel("Half-Difference Δ ∈ [0, 1]")
    ax2.set_ylabel("Absolute Value Distortion")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 0.25)
    ax2.legend(fontsize=7.2, loc="upper right")
    _style_ax(ax2)

    # Panel C: Stage-adaptive origin slope s'(0) across sorting stages
    eps = torch.tensor([1e-4], dtype=torch.float64)
    for idx, (lbl, rid, mname, n, kw) in enumerate([
        ("Fixed-Point m=1 (n=8)", "main_polysort_adaptive_n8_s42", "polysort_adaptive", 8, {}),
        ("Fixed-Point m=1 (n=16)", "main_polysort_adaptive_n16_s42", "polysort_adaptive", 16, {}),
        ("Unconstrained Odd A1 (n=8)", "abl_param_unconst_odd_n8_s42", "polysort_custom", 8,
         {"shared": False, "num_layers": 3, "num_terms": 1, "mode": "unconstrained_odd_cheby", "init": "cheon"}),
        ("Fixed-Point L=4 d=3 (n=8)", "abl_depth_L4_d3_n8_s42", "polysort_custom", 8,
         {"shared": False, "num_layers": 4, "num_terms": 1, "mode": "fixed_point_cheby", "init": "cheon"}),
    ]):
        m = _load_trained_model(rid, mname, n, **kw)
        slopes = []
        for st in m.sign_stages:
            with torch.no_grad():
                slopes.append(float(st(eps).item() / 1e-4))
        xs = np.linspace(0.0, 1.0, len(slopes))
        ax3.plot(xs, slopes, marker="o", lw=2.1, markersize=5, color=PALETTE[idx], label=lbl)

    ax3.axvline(0.60, color="#888888", ls="--", lw=1.0, alpha=0.65)
    ax3.text(0.61, 18.8, "Long-stride\nmerge reset", fontsize=7.2, color="#555555", va="top")
    ax3.set_title("(C) Stage-Adaptive Slope $s'(0)$ (Rediscovers Merge Strides)", fontsize=10.4, fontweight="bold")
    ax3.set_xlabel("Normalized Sorting Stage Progress (0 = First, 1 = Final)")
    ax3.set_ylabel("Comparator Origin Slope $s'(0)$")
    ax3.legend(fontsize=7.4, loc="upper left")
    _style_ax(ax3)

    fig.suptitle("Mechanistic Analysis: Why Learned Composite Polynomials Outperform Analytical CKKS & How Stages Specialize",
                 fontsize=12.0, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_both(fig, "fig3_mechanistic_comparators.png")


def build_fig4_output_examples() -> None:
    """Visual output examples highlighting BOTH positive wins (ID, Shift+100, Near-Ties) and negative failures."""
    rng = np.random.default_rng(2026)
    x_id = rng.uniform(-1.0, 1.0, size=(1, 8))
    x_shift = x_id + 100.0
    x_scale2 = x_id * 2.0

    models_spec = [
        ("PolySortNet-Adapt m=2\n(36p Pure Poly)", "main_polysort_adaptive_m2_n8_s42", "polysort_adaptive_m2", {}),
        ("Cheon f1^(3) Analytical\n(0p Pure Poly)", "ana_cheon_f1_L3_n8_s42", "analytical_cheon_f1", {"num_layers": 3}),
        ("PiNet-NCP-Skip\n(201k Pure Poly)", "main_pinet_ncpskip_n8_s42", "pinet_ncp_skip", {}),
        ("SetTransformer\n(96.6k Non-Poly)", "base_set_transformer_n8_s42", "set_transformer", {}),
        ("DiffSort-Cauchy\n(1p Non-Poly)", "base_diffsort_cauchy_n8_s42", "diffsort_cauchy", {}),
        ("Random Fixed Wiring\n(Negative Control)", "abl_wire_random_fixed_n8_s42", "polysort_custom",
         {"shared": False, "wiring_type": "random_fixed", "num_layers": 3, "num_terms": 1, "wiring_seed": 777}),
    ]
    loaded = [(lbl, _load_trained_model(rid, mname, 8, **kw)) for lbl, rid, mname, kw in models_spec]

    scenarios = [
        ("In-Distribution\nUniform [-1, 1]", x_id, (-1.15, 1.15)),
        ("Shift +100 OOD\n(Equivariance)", x_shift, (98.5, 101.5)),
        ("Scale 2.0× OOD\n(Blow-Up Cliff)", x_scale2, (-2.6, 2.6)),
    ]

    fig, axes = plt.subplots(3, 6, figsize=(16.5, 8.2))
    pos = np.arange(1, 9)
    for row, (sc_title, x_arr, ylim) in enumerate(scenarios):
        x_t = torch.from_numpy(x_arr).double()
        y_true = np.sort(x_arr[0])
        for col, (lbl, m) in enumerate(loaded):
            ax = axes[row, col]
            with torch.no_grad():
                y_pred = m(x_t).numpy()[0]
            finite = np.isfinite(y_pred).all()
            mae = float(np.mean(np.abs(y_pred - y_true))) if finite else float("inf")
            mono = bool(np.all(y_pred[:-1] <= y_pred[1:] + 1e-7)) if finite else False
            ax.scatter(pos, x_arr[0], color="#bbbbbb", marker="x", s=26, zorder=2, label="Input x")
            ax.plot(pos, y_true, "k-o", lw=1.6, ms=4, zorder=3, label="True sort(x)")
            if finite and np.max(np.abs(y_pred - np.mean(y_true))) < 3.0 * (ylim[1] - ylim[0]):
                y_plot = np.clip(y_pred, ylim[0] - 0.15 * (ylim[1] - ylim[0]), ylim[1] + 0.15 * (ylim[1] - ylim[0]))
                ax.plot(pos, y_plot, "--s", color=PALETTE[col % len(PALETTE)], lw=1.6, ms=4, zorder=4, label="Pred")
            else:
                msg = f"COLLAPSED\n(mean {np.mean(y_pred):.1f})" if (finite and np.max(np.abs(y_pred)) < 100) else "DIVERGED /\nOFF-SCALE"
                ax.text(4.5, np.mean(ylim), msg, ha="center", va="center",
                        fontsize=8.2, fontweight="bold", color="#CC0000",
                        bbox=dict(boxstyle="round,pad=0.25", fc="#FFEAEA", ec="#CC0000"))

            mae_txt = f"{mae:.4f}" if mae < 10 else (f"{mae:.1e}" if np.isfinite(mae) else "INF")
            ax.set_ylim(ylim)
            if row == 0:
                ax.set_title(f"{lbl}\nMAE={mae_txt}", fontsize=8.4, fontweight="bold")
            else:
                ax.set_title(f"MAE={mae_txt} | {'Mono' if mono else 'Inv'}", fontsize=8.2)
            if col == 0:
                ax.set_ylabel(sc_title, fontsize=8.5, fontweight="bold")
            if row == 2:
                ax.set_xlabel("Rank k", fontsize=8.0)
            _style_ax(ax)
            if row == 0 and col == 0:
                ax.legend(fontsize=6.8, loc="upper left")

    fig.suptitle("Concrete Output Examples: Positive Wins (ID & Shift +100) and Honest Negative Failures (Scale 2× & Random Wire)",
                 fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_both(fig, "fig4_output_examples.png")


def build_fig5_zeroshot_and_scale() -> None:
    """Two-panel OOD generalization figure: Zero-shot n=4..128 + Scale-extrapolation cliff s=0.1..5.0."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.7))

    # Left: Zero-shot length extrapolation
    zs_runs = [
        ("PolySortNet-Shared m=2 (6p Pure Poly)", "main_polysort_shared_m2_n8_s42", PALETTE[0], "-s"),
        ("PolySortNet-Shared m=1 (3p Pure Poly)", "main_polysort_shared_n8_s42", PALETTE[5], "-o"),
        ("Analytical Cheon f3∘g3 (0p Pure Poly)", "ana_cheon_fg_n8_s42", PALETTE[2], "--^"),
        ("Analytical Cheon f1^(3) (0p Pure Poly)", "ana_cheon_f1_L3_n8_s42", PALETTE[3], ":v"),
        ("SetTransformer (96.6k Non-Poly)", "base_set_transformer_n8_s42", PALETTE[4], "-.D"),
        ("DiffSort-Cauchy (1p Non-Poly)", "base_diffsort_cauchy_n8_s42", PALETTE[1], "--P"),
    ]
    for lbl, rid, c, fmt in zs_runs:
        d = _load_eval(rid)
        zs = d.get("zero_shot_length", {})
        if not zs:
            continue
        pairs_zs = sorted((int(str(k).replace("n=", "")), v["mae"]) for k, v in zs.items())
        xs = [p[0] for p in pairs_zs]
        ys = [p[1] for p in pairs_zs]
        ax1.plot(xs, ys, fmt, color=c, lw=2.1, ms=5.5, label=lbl)

    ax1.axvline(8, color="#555555", ls=":", lw=1.3, label="Training Length (n = 8)")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks([4, 8, 16, 32, 64, 128])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_ylim(1.8e-3, 0.75)
    ax1.set_xlabel("Evaluation List Length n (Zero-Shot up to 16× Training Length)", fontsize=9.6)
    ax1.set_ylabel("Held-Out Value MAE (log scale)", fontsize=9.6)
    ax1.set_title("(A) Zero-Shot List-Length Extrapolation (n = 4 → 128)", fontsize=10.8, fontweight="bold")
    ax1.legend(fontsize=7.4, loc="lower right", ncol=1)
    _style_ax(ax1)

    # Right: Scale extrapolation cliff
    scale_runs = [
        ("PolySortNet-Adapt L=4 d=3 (R_stab=1.5)", "abl_depth_L4_d3_n8_s42", PALETTE[0], "-s"),
        ("PolySortNet-Adapt m=1 (R_stab=1.3)", "main_polysort_adaptive_n8_s42", PALETTE[5], "-o"),
        ("PolySortNet-Adapt m=2 (R_stab=1.0)", "main_polysort_adaptive_m2_n8_s42", PALETTE[4], "-^"),
        ("A1 Unconstrained Odd (R_stab=1.1)", "abl_param_unconst_odd_n8_s42", PALETTE[3], ":x"),
        ("A6 Scale-Norm Wrapper (Non-Poly, R=5.0)", "abl_scale_norm_n8_s42", PALETTE[2], "--D"),
        ("DiffSort-Cauchy (Non-Poly, R=5.0)", "base_diffsort_cauchy_n8_s42", PALETTE[1], "--P"),
    ]
    scale_factors = [0.1, 0.5, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 5.0]
    key_map = {s: f"ood_scale_{int(s) if s == int(s) else s}" for s in scale_factors}
    for lbl, rid, c, fmt in scale_runs:
        d = _load_eval(rid)
        ood = d["ood"]
        ys = []
        for s in scale_factors:
            v = ood.get(key_map[s], {}).get("mae", float("inf"))
            rel = v / s if (np.isfinite(v) and v < 1e4) else 200.0
            ys.append(min(rel, 200.0))
        ax2.plot(scale_factors, ys, fmt, color=c, lw=2.0, ms=5.0, label=lbl)

    ax2.axvline(1.0, color="#555555", ls=":", lw=1.3)
    ax2.axhline(0.05, color="#CC0000", ls="--", lw=1.0, alpha=0.7, label="Stability Threshold (MAE/s = 0.05)")
    ax2.set_yscale("log")
    ax2.set_ylim(5e-4, 300.0)
    ax2.set_xlabel("Input Scale Multiplier s  (x_test = s · Uniform[-1, 1]⁸)", fontsize=9.6)
    ax2.set_ylabel("Relative Value Error (MAE / s, capped at 200)", fontsize=9.6)
    ax2.set_title("(B) Pure-Polynomial Scale Blow-Up Cliff vs. Non-Poly Saturation", fontsize=10.8, fontweight="bold")
    ax2.legend(fontsize=7.3, loc="lower right")
    _style_ax(ax2)

    fig.suptitle("Out-of-Distribution Generalization: Zero-Shot Length Scaling & Causal Scale-Stability Boundary",
                 fontsize=12.0, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_both(fig, "fig5_zeroshot_and_scale_cliff.png")


def build_fig6_ablations_and_training() -> None:
    """Four-panel causal ablation & training convergence figure."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6))

    # Panel A: Iso-Degree vs Iso-Parameter Depth Separation
    ax = axes[0, 0]
    specs_a = [
        ("Flat L=1, d=9\n(4p/st, D=9)", "abl_depth_L1_d9"),
        ("Comp L=2, d=3\n(2p/st, D=9)", "abl_depth_L2_d3"),
        ("Comp L=3, d=3\n(3p/st, D=27)", "main_polysort_adaptive"),
        ("Comp L=4, d=3\n(4p/st, D=81)", "abl_depth_L4_d3"),
    ]
    xs = np.arange(len(specs_a))
    w = 0.36
    m8 = [_load_eval(f"{p}_n8_s42")["test"]["mae"] for _, p in specs_a]
    m16 = [_load_eval(f"{p}_n16_s42")["test"]["mae"] for _, p in specs_a]
    ax.bar(xs - w / 2, m8, w, color=PALETTE[0], label="n = 8")
    ax.bar(xs + w / 2, m16, w, color=PALETTE[1], label="n = 16")
    for i, (v8, v16) in enumerate(zip(m8, m16)):
        ax.text(i - w / 2, v8 + 0.002, f"{v8:.3f}", ha="center", va="bottom", fontsize=7.8)
        ax.text(i + w / 2, v16 + 0.002, f"{v16:.3f}", ha="center", va="bottom", fontsize=7.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in specs_a], fontsize=8.4)
    ax.set_ylabel("Test Value MAE (↓)")
    ax.set_ylim(0, 0.098)
    ax.set_title("(A) A2 Depth Separation: Iso-Degree vs. Iso-Parameter (H3)", fontsize=10.3, fontweight="bold")
    ax.legend(fontsize=8.5)
    _style_ax(ax)

    # Panel B: A1 Parameterization Accuracy vs Scale Stability Trade-off
    ax = axes[0, 1]
    specs_b = [
        ("Fixed-Point Cheby (L=3)", "main_polysort_adaptive_n8_s42", PALETTE[0], (1.02, 0.0435)),
        ("Fixed-Point Cheby (L=4, Best Frontier)", "abl_depth_L4_d3_n8_s42", PALETTE[2], (0.72, 0.0085)),
        ("Unconstrained Odd Cheby (A1)", "abl_param_unconst_odd_n8_s42", PALETTE[1], (0.28, 0.0325)),
        ("Raw Monomial Basis (A1)", "abl_param_raw_monomial_n8_s42", PALETTE[3], (0.34, 0.0240)),
        ("General Cheby, No Odd Sym (A1)", "abl_param_general_cheby_n8_s42", PALETTE[4], (0.24, 0.0155)),
        ("Flat L=1 d=9 (Runge Cliff)", "abl_depth_L1_d9_n8_s42", PALETTE[7], (0.14, 0.0565)),
    ]
    for lbl, rid, c, xy_txt in specs_b:
        d = _load_eval(rid)
        ax.scatter(d["r_stable"], d["test"]["mae"], s=110, color=c, edgecolors="#111111", zorder=4)
        ax.annotate(
            lbl, (d["r_stable"], d["test"]["mae"]), xytext=xy_txt, fontsize=7.6,
            arrowprops=dict(arrowstyle="->", color="#666666", lw=0.9, shrinkA=2, shrinkB=5),
        )
    ax.set_xlabel("Upper Scale Stability Radius R_stable (↑ better)", fontsize=9.2)
    ax.set_ylabel("Held-Out Test MAE at n = 8 (↓ better)", fontsize=9.2)
    ax.set_xlim(-0.12, 1.68)
    ax.set_ylim(0.0, 0.066)
    ax.set_title("(B) A1 Accuracy vs. Stability Frontier (H6 Part 1 Falsified)", fontsize=10.3, fontweight="bold")
    _style_ax(ax)

    # Panel C: A4 Wiring Topology Causal Impact
    ax = axes[1, 0]
    specs_c = [
        ("Batcher Odd-Even\nMerge", "abl_wire_oddeven_merge"),
        ("Bitonic\n(Default)", "main_polysort_adaptive"),
        ("Habermann Odd-Even\nTransposition", "abl_wire_oddeven_trans"),
        ("Random Fixed\n(Neg. Control)", "abl_wire_random_fixed"),
    ]
    xs = np.arange(len(specs_c))
    maes_c = [_load_eval(f"{p}_n16_s42")["test"]["mae"] for _, p in specs_c]
    monos_c = [_load_eval(f"{p}_n16_s42")["test"]["mono_rate"] * 100.0 for _, p in specs_c]
    ax_twin = ax.twinx()
    b1 = ax.bar(xs - 0.18, maes_c, 0.35, color=PALETTE[0], label="MAE at n=16 (↓)")
    b2 = ax_twin.bar(xs + 0.18, monos_c, 0.35, color=PALETTE[2], alpha=0.85, label="Monotonicity % (↑)")
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in specs_c], fontsize=8.3)
    ax.set_ylabel("Test Value MAE (n = 16)", color=PALETTE[0])
    ax_twin.set_ylabel("Full-List Monotonicity % (n = 16)", color=PALETTE[2])
    ax_twin.set_ylim(0, 112)
    ax.set_title("(C) A4 Sorting-Network Wiring Topology (n = 16, H6 Part 2)", fontsize=10.3, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    lines = [b1, b2]
    ax.legend(lines, ["Value MAE (↓)", "Monotonicity % (↑)"], loc="upper left", fontsize=8.0)

    # Panel D: Training Convergence Curves (n = 16)
    ax = axes[1, 1]
    curve_specs = [
        ("PolySortNet-Adapt m=2 (60p)", "main_polysort_adaptive_m2_n16_s42", PALETTE[0]),
        ("PolySortNet-Adapt L=4 (40p)", "abl_depth_L4_d3_n16_s42", PALETTE[2]),
        ("PolySortNet-Adapt m=1 (30p)", "main_polysort_adaptive_n16_s42", PALETTE[5]),
        ("DeepSets-Poly (131k)", "main_deepsets_poly_n16_s42", PALETTE[3]),
        ("PiNet-NCP-Skip (206k)", "main_pinet_ncpskip_n16_s42", PALETTE[4]),
        ("SetTransformer (96.6k Non-Poly)", "base_set_transformer_n16_s42", PALETTE[1]),
    ]
    for lbl, rid, c in curve_specs:
        lp = RESULTS_DIR / "runs" / rid / "train_log.jsonl"
        if not lp.exists():
            continue
        rows = [json.loads(line) for line in lp.read_text(encoding="utf-8").splitlines() if line.strip()]
        ax.plot([r["epoch"] for r in rows], [r["val_mae"] for r in rows], lw=2.0, color=c, label=lbl)
    ax.set_yscale("log")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Validation Value MAE (log scale)")
    ax.set_title("(D) Validation Convergence Trajectories (n = 16, Seed 42)", fontsize=10.3, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper right")
    _style_ax(ax)

    fig.suptitle("Causal Ablations (A1–A4) and Training Dynamics", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_both(fig, "fig6_ablations_and_training.png")


def export_web_assets() -> None:
    """Export client-side pure-polynomial JSON weights + empirical explorer dataset for GitHub Pages."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    web_models = [
        ("polysort_shared_m2_n8", "PolySortNet-Shared m=2 (6 params, zero-shot any n)", "main_polysort_shared_m2_n8_s42", "polysort_shared_m2", 8, {}),
        ("polysort_shared_m1_n8", "PolySortNet-Shared m=1 (3 params, zero-shot any n)", "main_polysort_shared_n8_s42", "polysort_shared", 8, {}),
        ("polysort_adaptive_m2_n8", "PolySortNet-Adaptive m=2 (36 params, n=8)", "main_polysort_adaptive_m2_n8_s42", "polysort_adaptive_m2", 8, {}),
        ("polysort_adaptive_m2_n16", "PolySortNet-Adaptive m=2 (60 params, n=16)", "main_polysort_adaptive_m2_n16_s42", "polysort_adaptive_m2", 16, {}),
        ("polysort_adaptive_L4_n8", "PolySortNet-Adaptive L=4 d=3 (24 params, n=8, R_stab=1.5)", "abl_depth_L4_d3_n8_s42", "polysort_custom", 8,
         {"shared": False, "num_layers": 4, "num_terms": 1, "mode": "fixed_point_cheby", "init": "cheon"}),
        ("analytical_cheon_fg", "Analytical Cheon f3∘g3 (0 params, CKKS baseline)", "ana_cheon_fg_n8_s42", "analytical_cheon_fg", 8, {}),
        ("analytical_cheon_f1", "Analytical Cheon f1^(3) (0 params, CKKS baseline)", "ana_cheon_f1_L3_n8_s42", "analytical_cheon_f1", 8, {"num_layers": 3}),
        ("analytical_remez_L3", "Analytical HiGHS Remez L=3 (0 params, CKKS baseline)", "ana_remez_L3_n8_s42", "analytical_remez", 8, {"num_layers": 3, "num_terms": 1, "delta": 0.10}),
    ]
    bundle: Dict[str, Any] = {}
    for key, label, rid, mname, n, kw in web_models:
        m = _load_trained_model(rid, mname, n, **kw)
        ev = _load_eval(rid)
        tmp = DOCS_DIR / "_tmp_w.json"
        payload = export_browser_poly_weights(
            m, rid, tmp,
            metadata={
                "label": label,
                "mae": ev["test"]["mae"],
                "mono_rate": ev["test"]["mono_rate"],
                "exact_sort_0.05": ev["test"]["exact_sort_0.05"],
                "r_stable": ev["r_stable"],
                "n_params": ev["n_params"],
            },
        )
        if tmp.exists():
            tmp.unlink()
        bundle[key] = payload
    (DOCS_DIR / "weights.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    # Also bundle empirical tables for interactive web explorer
    ms = json.loads((RESULTS_DIR / "multiseed_summary.json").read_text(encoding="utf-8")) if (RESULTS_DIR / "multiseed_summary.json").exists() else {}
    st = json.loads((RESULTS_DIR / "statistical_tests.json").read_text(encoding="utf-8")) if (RESULTS_DIR / "statistical_tests.json").exists() else []
    abls = []
    for p in sorted((RESULTS_DIR / "runs").glob("abl_*/eval_report.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        abls.append({
            "run_id": d["run_id"],
            "n": d["n"],
            "n_params": d["n_params"],
            "is_pure_polynomial": d["is_pure_polynomial"],
            "mae": d["test"]["mae"],
            "mono_rate": d["test"]["mono_rate"],
            "exact_sort_0.05": d["test"]["exact_sort_0.05"],
            "kendall_tau": d["test"]["kendall_tau"],
            "r_stable": d["r_stable"],
            "shift100_mae": d["ood"]["ood_shift_100"]["mae"],
        })
    (DOCS_DIR / "explorer_data.json").write_text(
        json.dumps({"multiseed": ms, "statistical_tests": st, "ablations": abls}, indent=2),
        encoding="utf-8",
    )


def build_all_assets() -> None:
    build_fig1_architecture()
    build_fig2_main_comparison()
    build_fig3_mechanistic_comparators()
    build_fig4_output_examples()
    build_fig5_zeroshot_and_scale()
    build_fig6_ablations_and_training()
    export_web_assets()
    print("Successfully built all 6 publication figures and GitHub Pages explorer payloads.")


if __name__ == "__main__":
    build_all_assets()
