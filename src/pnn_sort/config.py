"""Typed configuration and shared constants for pnn-sort."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

DEFAULT_SEEDS: tuple[int, ...] = (42, 123, 456, 789, 1024)
SCALE_SWEEP_FACTORS: tuple[float, ...] = (0.1, 0.5, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 5.0)
SHIFT_SWEEP_OFFSETS: tuple[float, ...] = (1.0, 10.0, 100.0)
ZERO_SHOT_LENGTHS: tuple[int, ...] = (4, 8, 16, 32, 64, 128)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainConfig:
    run_id: str
    model_name: str
    n: int = 8
    seed: int = 42
    train_size: int = 200_000
    val_size: int = 10_000
    test_size: int = 10_000
    ood_size: int = 2_000
    epochs: int = 60
    batch_size: int = 512
    lr: float = 3e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    loss_name: str = "mse"
    mono_weight: float = 0.0
    w1_weight: float = 0.0
    shuffled_targets: bool = False
    dtype: str = "float32"
    device: str = field(default_factory=default_device)
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    resume: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
