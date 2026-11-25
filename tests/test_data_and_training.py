"""Data leakage, checkpoint resume determinism, statistical test, dashboard, and pipeline smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
import urllib.request

import numpy as np
import torch

from pnn_sort.config import TrainConfig
from pnn_sort.dashboard import start_dashboard_thread
from pnn_sort.data import generate_ood_suites, generate_split, verify_disjoint_splits
from pnn_sort.eval import bootstrap_ci, holm_bonferroni, paired_wilcoxon_test
from pnn_sort.pipeline import run_experiment
from pnn_sort.train import train_run


def test_zero_data_leakage_across_splits() -> None:
    x_tr, y_tr = generate_split("train", n=8, size=5000, seed=42)
    x_va, y_va = generate_split("val", n=8, size=2000, seed=42)
    x_te, y_te = generate_split("test", n=8, size=2000, seed=9999)
    assert verify_disjoint_splits(x_tr, x_va, x_te)
    assert torch.allclose(y_tr, torch.sort(x_tr, dim=-1).values)
    assert torch.allclose(y_te, torch.sort(x_te, dim=-1).values)

    ood = generate_ood_suites(n=8, size=100, seed=9999)
    assert "ood_nearties_1e-3" in ood and "ood_scale_5" in ood and "ood_shift_100" in ood


def test_checkpoint_resume_determinism(tmp_path: Path) -> None:
    # Run 4 epochs straight through vs 2 epochs + resume to 4 epochs -> exact weight match!
    cfg_full = TrainConfig(
        run_id="resume_full",
        model_name="polysort_adaptive",
        n=8,
        seed=42,
        train_size=1024,
        val_size=256,
        epochs=4,
        batch_size=256,
        device="cpu",
        resume=False,
    )
    m_full, _ = train_run(cfg_full, checkpoints_root=tmp_path / "c1", results_root=tmp_path / "r1")

    cfg_p1 = TrainConfig(**{**cfg_full.to_dict(), "run_id": "resume_split", "resume": False})
    train_run(cfg_p1, checkpoints_root=tmp_path / "c2", results_root=tmp_path / "r2", stop_after_epoch=2)

    cfg_p2 = TrainConfig(**{**cfg_full.to_dict(), "run_id": "resume_split", "resume": True})
    m_resumed, _ = train_run(cfg_p2, checkpoints_root=tmp_path / "c2", results_root=tmp_path / "r2")

    for p1, p2 in zip(m_full.parameters(), m_resumed.parameters()):
        assert torch.allclose(p1, p2, atol=1e-6), "Resumed training diverged from uninterrupted run!"


def test_statistical_rigor_utilities() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.01, 0.002, size=500)
    b = rng.normal(0.02, 0.002, size=500)
    mean, lo, hi = bootstrap_ci(a, num_resamples=2000, seed=1)
    assert lo < mean < hi
    w = paired_wilcoxon_test(a, b, num_boot=2000, seed=1)
    assert w["p_value"] < 1e-10 and w["mean_diff"] < 0 and w["rank_biserial"] < -0.9
    adj = holm_bonferroni([0.001, 0.02, 0.04, 0.2])
    assert adj[0] == 0.004 and adj[1] == 0.06 and adj[3] == 0.2


def test_end_to_end_pipeline_and_dashboard_smoke(tmp_path: Path) -> None:
    cfg = TrainConfig(
        run_id="smoke_polysort",
        model_name="polysort_shared",
        n=8,
        seed=42,
        train_size=512,
        val_size=128,
        test_size=128,
        ood_size=64,
        epochs=2,
        batch_size=128,
        device="cpu",
        resume=False,
    )
    # Smoke run with temporary roots via train_run + dashboard check
    train_run(cfg, checkpoints_root=tmp_path / "ckpts", results_root=tmp_path / "res")
    server = start_dashboard_thread(port=0, results_root=tmp_path / "res")
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs", timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert len(payload["runs"]) == 1 and payload["runs"][0]["run_id"] == "smoke_polysort"
    finally:
        server.shutdown()
