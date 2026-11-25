"""Top-level modular pipeline wrapper orchestrating audit, training, evaluation, and reporting.

Usage example:
    uv run python -m pnn_sort.pipeline --run-id main_polysort_shared_n8_s42 --model polysort_shared --n 8 --seed 42
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

import torch

from pnn_sort.audit import audit_pure_polynomial
from pnn_sort.config import TrainConfig
from pnn_sort.eval import evaluate_model_full
from pnn_sort.models import build_model
from pnn_sort.train import train_run


def run_experiment(cfg: TrainConfig, eval_zero_shot: bool = True, eval_float64: bool = True) -> Dict[str, Any]:
    """Execute one complete, auditable experiment: Purity Audit -> Train -> Evaluate."""
    # Phase 1: Mechanical Pure-Polynomial FX Audit (pre-training gate)
    probe = build_model(cfg.model_name, n=cfg.n, **cfg.model_kwargs)
    if getattr(probe, "is_pure_polynomial", False):
        audit_res = audit_pure_polynomial(probe)
        if not audit_res.passed:
            raise RuntimeError(f"Pure-polynomial FX audit FAILED for {cfg.model_name}: {audit_res.violations}")

    # Phase 2: Resumable training (or closed-form / zero-training evaluation)
    model, train_summary = train_run(cfg)

    # Phase 3: Full held-out ID + OOD + float64 + zero-shot length evaluation
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    eval_report = evaluate_model_full(
        model=model,
        run_id=cfg.run_id,
        model_name=cfg.model_name,
        n=cfg.n,
        test_size=cfg.test_size,
        ood_size=cfg.ood_size,
        seed=cfg.seed,
        device=cfg.device,
        dtype=dtype,
        eval_zero_shot=eval_zero_shot,
        eval_float64=eval_float64,
    )
    return {"train": train_summary, "eval": eval_report}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single pnn-sort experiment end-to-end.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", default="mse")
    parser.add_argument("--mono-weight", type=float, default=0.0)
    parser.add_argument("--w1-weight", type=float, default=0.0)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--shuffled-targets", action="store_true")
    parser.add_argument("--model-kwargs", default="{}", help="JSON dict of model keyword args")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig(
        run_id=args.run_id,
        model_name=args.model,
        n=args.n,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_name=args.loss,
        mono_weight=args.mono_weight,
        w1_weight=args.w1_weight,
        dtype=args.dtype,
        shuffled_targets=args.shuffled_targets,
        model_kwargs=json.loads(args.model_kwargs),
        resume=not args.no_resume,
    )
    out = run_experiment(cfg)
    t = out["eval"]["test"]
    print(
        f"[{cfg.run_id}] MAE={t['mae']:.6f} | RMSE={t['rmse']:.6f} | "
        f"Mono={t['mono_rate']*100:.2f}% | PairAcc={t['pair_acc']*100:.2f}% | "
        f"Tau={t['kendall_tau']:.5f} | R_stable={out['eval']['r_stable']}"
    )


if __name__ == "__main__":
    main()
