"""Publishing utilities: export pure-polynomial browser weights and sync to GitHub / Hugging Face."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict

import torch

from pnn_sort.config import CHECKPOINTS_DIR, PROJECT_ROOT, RESULTS_DIR
from pnn_sort.models import PolyPairwiseRank, PolySortNet


def export_browser_poly_weights(
    model: torch.nn.Module,
    run_id: str,
    out_path: Path,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Export trained pure-polynomial comparator coefficients to JSON for client-side JS execution."""
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "model_class": model.__class__.__name__,
        "n": getattr(model, "n", None),
        "metadata": metadata or {},
    }
    if isinstance(model, PolySortNet):
        payload["wiring_type"] = model.wiring_type
        payload["shared"] = model.shared
        payload["mode"] = model.mode
        payload["num_layers"] = model.num_layers
        payload["num_terms"] = model.num_terms
        if model.sign_shared is not None:
            payload["coeffs_shared"] = model.sign_shared.coeffs.detach().cpu().double().tolist()
        else:
            payload["coeffs_stages"] = [
                s.coeffs.detach().cpu().double().tolist() for s in model.sign_stages  # type: ignore[union-attr]
            ]
    elif isinstance(model, PolyPairwiseRank):
        payload["cmp_coeffs"] = model.cmp_sign.coeffs.detach().cpu().double().tolist()
        payload["ind_coeffs"] = model.ind_sign.coeffs.detach().cpu().double().tolist()
    else:
        raise TypeError(f"Browser weight export supports PolySortNet/PolyPairwiseRank, got {type(model)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def git_commit_and_push(message: str, cwd: Path = PROJECT_ROOT) -> str:
    """Stage tracked project artifacts, commit if dirty, and push to GitHub ``origin/main``."""
    subprocess.run(["git", "add", "-A"], cwd=cwd, check=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True)
    if status.stdout.strip():
        subprocess.run(["git", "commit", "-m", message], cwd=cwd, check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=cwd, check=True)
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True)
    return rev.stdout.strip()


def upload_to_huggingface(
    repo_id: str = "BurnyCoder/polynomial-neural-network-sort-list",
    include_checkpoints: bool = True,
) -> None:
    """Upload README, reports, results, and checkpoints cleanly to Hugging Face Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    # Upload lightweight root docs + reports + results
    for rel in ("README.md",):
        p = PROJECT_ROOT / rel
        if p.exists():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=rel, repo_id=repo_id, repo_type="model")
    for folder in ("reports", "results"):
        d = PROJECT_ROOT / folder
        if d.exists():
            api.upload_folder(
                folder_path=str(d),
                path_in_repo=folder,
                repo_id=repo_id,
                repo_type="model",
                ignore_patterns=["*.tmp", "*.npz"],
            )
    if include_checkpoints and CHECKPOINTS_DIR.exists():
        api.upload_folder(
            folder_path=str(CHECKPOINTS_DIR),
            path_in_repo="checkpoints",
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=["*/best.pt"],
        )
