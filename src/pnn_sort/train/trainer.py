"""Resumable GPU/CPU training engine with atomic checkpointing and live JSONL telemetry."""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from pnn_sort.config import CHECKPOINTS_DIR, RESULTS_DIR, TrainConfig
from pnn_sort.data import generate_split
from pnn_sort.models import LinearControl, build_model, count_parameters
from .losses import sorting_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _quick_val_metrics(model: nn.Module, x_val: torch.Tensor, y_val: torch.Tensor, batch_size: int = 2048) -> Dict[str, float]:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, x_val.shape[0], batch_size):
            preds.append(model(x_val[i : i + batch_size]))
    p = torch.cat(preds, dim=0)
    if not torch.isfinite(p).all():
        return {"val_mae": float("inf"), "val_mse": float("inf"), "val_mono_rate": 0.0, "val_pair_acc": 0.0}
    diff = p - y_val
    mae = float(diff.abs().mean().item())
    mse = float((diff * diff).mean().item())
    mono = float(((p[:, :-1] <= p[:, 1:] + 1e-7).all(dim=-1)).float().mean().item())
    # Fast vectorized pairwise order accuracy: fraction of pairs (i < j) with p_i <= p_j
    n = p.shape[1]
    tri = torch.triu(torch.ones(n, n, dtype=torch.bool, device=p.device), diagonal=1)
    pair_ok = (p.unsqueeze(2) <= p.unsqueeze(1) + 1e-7)[:, tri]
    pair_acc = float(pair_ok.float().mean().item())
    return {
        "val_mae": mae,
        "val_mse": mse,
        "val_mono_rate": mono,
        "val_pair_acc": pair_acc,
    }


def train_run(
    cfg: TrainConfig,
    checkpoints_root: Path = CHECKPOINTS_DIR,
    results_root: Path = RESULTS_DIR,
    stop_after_epoch: int | None = None,
) -> Tuple_Model_Summary:
    """Train (or resume) a model according to ``cfg``, returning ``(model, history_summary)``."""
    seed_everything(cfg.seed)
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    device = torch.device(cfg.device)

    run_ckpt_dir = checkpoints_root / cfg.run_id
    run_res_dir = results_root / "runs" / cfg.run_id
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_res_dir.mkdir(parents=True, exist_ok=True)

    last_ckpt_path = run_ckpt_dir / "last.pt"
    best_ckpt_path = run_ckpt_dir / "best.pt"
    log_path = run_res_dir / "train_log.jsonl"
    live_state_path = results_root / "live_dashboard_state.json"

    model = build_model(cfg.model_name, n=cfg.n, **cfg.model_kwargs).to(device=device, dtype=dtype)
    n_params = count_parameters(model)

    x_train, y_train = generate_split(
        "train", cfg.n, cfg.train_size, seed=cfg.seed, dtype=dtype, shuffled_targets=cfg.shuffled_targets
    )
    x_val, y_val = generate_split("val", cfg.n, cfg.val_size, seed=cfg.seed, dtype=dtype, shuffled_targets=False)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    t0 = time.perf_counter()

    # Zero-training or closed-form models
    if isinstance(model, LinearControl):
        model.fit_closed_form(x_train, y_train)
        val_m = _quick_val_metrics(model, x_val, y_val)
        _atomic_torch_save({"model": model.state_dict(), "config": cfg.to_dict(), "val_metrics": val_m}, best_ckpt_path)
        summary = {
            "run_id": cfg.run_id,
            "model_name": cfg.model_name,
            "n": cfg.n,
            "seed": cfg.seed,
            "n_params": n_params,
            "epochs_completed": 0,
            "best_val_mae": val_m["val_mae"],
            "best_val_mono_rate": val_m["val_mono_rate"],
            "best_val_pair_acc": val_m["val_pair_acc"],
            "diverged": False,
            "train_seconds": time.perf_counter() - t0,
        }
        (run_res_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return model, summary

    if n_params == 0 or cfg.epochs <= 0:
        val_m = _quick_val_metrics(model, x_val, y_val)
        _atomic_torch_save({"model": model.state_dict(), "config": cfg.to_dict(), "val_metrics": val_m}, best_ckpt_path)
        summary = {
            "run_id": cfg.run_id,
            "model_name": cfg.model_name,
            "n": cfg.n,
            "seed": cfg.seed,
            "n_params": n_params,
            "epochs_completed": 0,
            "best_val_mae": val_m["val_mae"],
            "best_val_mono_rate": val_m["val_mono_rate"],
            "best_val_pair_acc": val_m["val_pair_acc"],
            "diverged": not math.isfinite(val_m["val_mae"]),
            "train_seconds": time.perf_counter() - t0,
        }
        (run_res_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return model, summary

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(cfg.train_size / cfg.batch_size)
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    start_epoch = 0
    best_val_mae = float("inf")
    best_val_metrics: Dict[str, float] = {}
    diverged = False

    if cfg.resume and last_ckpt_path.exists():
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_mae = float(ckpt["best_val_mae"])
        best_val_metrics = dict(ckpt.get("best_val_metrics", {}))
        diverged = bool(ckpt.get("diverged", False))
        if "rng_torch" in ckpt:
            torch.set_rng_state(ckpt["rng_torch"].cpu())
        if "rng_cuda" in ckpt and torch.cuda.is_available() and ckpt["rng_cuda"] is not None:
            torch.cuda.set_rng_state(ckpt["rng_cuda"].cpu())
        if "rng_np" in ckpt:
            np.random.set_state(ckpt["rng_np"])
        if "rng_py" in ckpt:
            random.setstate(ckpt["rng_py"])
    else:
        if log_path.exists():
            log_path.unlink()

    n_samples = x_train.shape[0]
    end_epoch = cfg.epochs if stop_after_epoch is None else min(cfg.epochs, stop_after_epoch)
    for epoch in range(start_epoch, end_epoch):
        if diverged:
            break
        model.train()
        perm = torch.randperm(n_samples, device=device)
        running_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            xb = x_train[idx]
            yb = y_train[idx]
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = sorting_loss(
                pred,
                yb,
                loss_name=cfg.loss_name,
                mono_weight=cfg.mono_weight,
                w1_weight=cfg.w1_weight,
            )
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()
            if cfg.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        val_m = _quick_val_metrics(model, x_val, y_val)
        if not math.isfinite(val_m["val_mae"]) or val_m["val_mae"] > 1e6:
            diverged = True

        if not diverged and val_m["val_mae"] < best_val_mae:
            best_val_mae = val_m["val_mae"]
            best_val_metrics = dict(val_m)
            _atomic_torch_save(
                {"model": model.state_dict(), "config": cfg.to_dict(), "epoch": epoch, "val_metrics": val_m},
                best_ckpt_path,
            )

        rec = {
            "run_id": cfg.run_id,
            "model_name": cfg.model_name,
            "n": cfg.n,
            "seed": cfg.seed,
            "epoch": epoch + 1,
            "total_epochs": cfg.epochs,
            "lr": float(scheduler.get_last_lr()[0]),
            "train_loss": train_loss,
            **val_m,
            "best_val_mae": best_val_mae,
            "diverged": diverged,
            "elapsed_sec": time.perf_counter() - t0,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        # Update live dashboard state file atomically
        tmp_live = live_state_path.with_suffix(".json.tmp")
        tmp_live.write_text(json.dumps(rec), encoding="utf-8")
        os.replace(tmp_live, live_state_path)

        # Save resumable checkpoint
        _atomic_torch_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_mae": best_val_mae,
                "best_val_metrics": best_val_metrics,
                "diverged": diverged,
                "config": cfg.to_dict(),
                "rng_torch": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                "rng_np": np.random.get_state(),
                "rng_py": random.getstate(),
            },
            last_ckpt_path,
        )

    # Restore best checkpoint if available
    if best_ckpt_path.exists():
        best_state = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_state["model"])
    else:
        best_val_metrics = _quick_val_metrics(model, x_val, y_val)

    summary = {
        "run_id": cfg.run_id,
        "model_name": cfg.model_name,
        "n": cfg.n,
        "seed": cfg.seed,
        "n_params": n_params,
        "epochs_completed": cfg.epochs if not diverged else start_epoch,
        "best_val_mae": best_val_mae,
        "best_val_mono_rate": best_val_metrics.get("val_mono_rate", 0.0),
        "best_val_pair_acc": best_val_metrics.get("val_pair_acc", 0.0),
        "diverged": diverged,
        "train_seconds": time.perf_counter() - t0,
    }
    (run_res_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return model, summary


Tuple_Model_Summary = tuple[nn.Module, Dict[str, Any]]
