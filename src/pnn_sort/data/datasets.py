"""Seed-isolated dataset generation and OOD stress suites.

All splits derive independent NumPy PCG64 generators from SHA-256 hashes of
``(split_name, n, seed)`` so train/val/test/OOD sets are strictly isolated and
100% reproducible across platforms.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Tuple

import numpy as np
import torch

from pnn_sort.config import SCALE_SWEEP_FACTORS, SHIFT_SWEEP_OFFSETS


def make_rng(split_name: str, n: int, seed: int) -> np.random.Generator:
    """Create an independent deterministic NumPy RNG keyed by split, length, and seed."""
    digest = hashlib.sha256(f"pnn-sort::{split_name}::n={n}::seed={seed}".encode("utf-8")).digest()
    seed_int = int.from_bytes(digest[:16], byteorder="little", signed=False)
    return np.random.default_rng(seed_int)


def generate_split(
    split_name: str,
    n: int,
    size: int,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
    shuffled_targets: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate Uniform[-1, 1]^n inputs and their exact non-decreasing sorted targets.

    If ``shuffled_targets=True`` (used solely for the leakage sanity control), targets
    are replaced with independent random permutations of each input row.
    """
    rng = make_rng(split_name, n, seed)
    x_np = rng.uniform(-1.0, 1.0, size=(size, n))
    if shuffled_targets:
        perm_rng = make_rng(f"{split_name}_shuf_target", n, seed)
        y_np = np.empty_like(x_np)
        for i in range(size):
            y_np[i] = x_np[i, perm_rng.permutation(n)]
    else:
        y_np = np.sort(x_np, axis=-1)
    x = torch.from_numpy(x_np).to(dtype=dtype)
    y = torch.from_numpy(y_np).to(dtype=dtype)
    return x, y


def _make_nearties(rng: np.random.Generator, size: int, n: int, delta: float) -> np.ndarray:
    """Plant adjacent pairs separated by exact gap ``delta``, then randomly permute."""
    x = rng.uniform(-0.95, 0.95, size=(size, n))
    num_pairs = max(1, n // 2)
    for p in range(num_pairs):
        a = 2 * p
        b = 2 * p + 1
        if b < n:
            x[:, b] = np.clip(x[:, a] + delta, -1.0, 1.0)
    # Permute each row randomly so planted pairs appear at arbitrary wire positions.
    keys = rng.random((size, n))
    order = np.argsort(keys, axis=-1)
    return np.take_along_axis(x, order, axis=-1)


def generate_ood_suites(
    n: int,
    size: int = 2_000,
    seed: int = 9999,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Generate all 10 preregistered OOD & stress test suites."""
    suites: Dict[str, np.ndarray] = {}

    # 1. Truncated Gaussian
    rng = make_rng("ood_gaussian", n, seed)
    suites["ood_gaussian"] = np.clip(rng.normal(0.0, 0.35, size=(size, n)), -1.0, 1.0)

    # 2. Bimodal mixture
    rng = make_rng("ood_bimodal", n, seed)
    modes = rng.choice([-0.7, 0.7], size=(size, n))
    suites["ood_bimodal"] = np.clip(modes + rng.normal(0.0, 0.10, size=(size, n)), -1.0, 1.0)

    # 3. Skewed Beta(0.5, 5.0) mapped to [-1, 1]
    rng = make_rng("ood_skewed", n, seed)
    suites["ood_skewed"] = 2.0 * rng.beta(0.5, 5.0, size=(size, n)) - 1.0

    # 4. Discrete quantized integers in {-10, ..., 10} / 10 (frequent exact ties)
    rng = make_rng("ood_integers", n, seed)
    suites["ood_integers"] = rng.integers(-10, 11, size=(size, n)) / 10.0

    # 5. Exact duplicates (50% repeated elements, randomly permuted)
    rng = make_rng("ood_duplicates_50", n, seed)
    half = max(1, n // 2)
    base = rng.uniform(-1.0, 1.0, size=(size, half))
    dup = np.concatenate([base, base[:, : n - half]], axis=-1)
    order = np.argsort(rng.random((size, n)), axis=-1)
    suites["ood_duplicates_50"] = np.take_along_axis(dup, order, axis=-1)

    # 6. Adversarial near-ties across 3 gap scales
    for label, delta in [("1e-2", 1e-2), ("1e-3", 1e-3), ("1e-4", 1e-4)]:
        rng = make_rng(f"ood_nearties_{label}", n, seed)
        suites[f"ood_nearties_{label}"] = _make_nearties(rng, size, n, delta)

    # 7. Edge cases: already sorted and strictly reverse-sorted
    rng = make_rng("ood_edge_sorted", n, seed)
    edge_base = np.sort(rng.uniform(-1.0, 1.0, size=(size, n)), axis=-1)
    suites["ood_edge_sorted"] = edge_base
    suites["ood_edge_reversed"] = edge_base[:, ::-1].copy()

    # 8. Arbitrary constant shifts x + c*1
    rng_shift = make_rng("ood_shift_base", n, seed)
    shift_base = rng_shift.uniform(-1.0, 1.0, size=(size, n))
    for c in SHIFT_SWEEP_OFFSETS:
        suites[f"ood_shift_{int(c)}"] = shift_base + c

    # 9. Scale extrapolation sweep s * x
    rng_scale = make_rng("ood_scale_base", n, seed)
    scale_base = rng_scale.uniform(-1.0, 1.0, size=(size, n))
    for s in SCALE_SWEEP_FACTORS:
        suites[f"ood_scale_{s:g}"] = scale_base * s

    out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name, x_np in suites.items():
        y_np = np.sort(x_np, axis=-1)
        out[name] = (
            torch.from_numpy(x_np).to(dtype=dtype),
            torch.from_numpy(y_np).to(dtype=dtype),
        )
    return out


def dataset_row_hashes(x: torch.Tensor) -> set[bytes]:
    """Compute exact byte hashes of rows to verify zero train/val/test overlap."""
    arr = np.ascontiguousarray(x.detach().cpu().numpy())
    return {hashlib.blake2b(row.tobytes(), digest_size=16).digest() for row in arr}


def verify_disjoint_splits(*tensors: torch.Tensor) -> bool:
    """Return True iff all provided datasets share zero exact rows."""
    seen: set[bytes] = set()
    for t in tensors:
        h = dataset_row_hashes(t)
        if seen.intersection(h):
            return False
        seen.update(h)
    return True
