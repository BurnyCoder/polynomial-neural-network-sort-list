"""Sorting-network wiring matrices (Bitonic, Odd-Even Merge, Odd-Even Transposition, Random).

Matrix representation ``(split_a, split_b, combine_min, combine_max)`` follows
Petersen et al. (ICML 2021 / ICLR 2022, ``github.com/Felix-Petersen/diffsort``),
extended with Batcher's Odd-Even Mergesort (Batcher, 1968) and a fixed-depth
random-pair negative control.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch

WiringStage = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _pairs_to_stage_matrices(n: int, pairs: List[Tuple[int, int]]) -> WiringStage:
    """Convert a list of disjoint oriented comparator pairs ``(low_idx, high_idx)`` into matrices.

    Every wire not participating in a pair receives an identity pass-through gate
    ``(i, i)`` with combine weights ``(0.5, 0.5)`` so ``min(x_i, x_i) = max(x_i, x_i) = x_i``
    identically for any odd-symmetric comparator (exact sum conservation & shift equivariance).
    """
    used = set()
    for a, b in pairs:
        if a in used or b in used or a == b:
            raise ValueError(f"Overlapping or degenerate comparator pair ({a}, {b}) in stage")
        used.add(a)
        used.add(b)
    passthrough = [i for i in range(n) if i not in used]
    k = len(pairs) + len(passthrough)
    split_a = np.zeros((k, n), dtype=np.float64)
    split_b = np.zeros((k, n), dtype=np.float64)
    combine_min = np.zeros((n, k), dtype=np.float64)
    combine_max = np.zeros((n, k), dtype=np.float64)

    idx = 0
    for a, b in pairs:
        split_a[idx, a] = 1.0
        split_b[idx, b] = 1.0
        combine_min[a, idx] = 1.0
        combine_max[b, idx] = 1.0
        idx += 1
    for wire in passthrough:
        split_a[idx, wire] = 1.0
        split_b[idx, wire] = 1.0
        combine_min[wire, idx] = 0.5
        combine_max[wire, idx] = 0.5
        idx += 1
    return split_a, split_b, combine_min, combine_max


def bitonic_network(n: int) -> List[WiringStage]:
    """Batcher's bitonic sorting network (Batcher 1968; Petersen et al. 2021).

    Supports arbitrary ``n >= 2`` by pruning top wires from the smallest enclosing
    power of two ``2**ceil(log2(n))``. Depth = ``p*(p+1)/2`` where ``p = ceil(log2(n))``.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    num_blocks = math.ceil(math.log2(n))
    m_pow = 2**num_blocks
    offset = m_pow - n
    stages: List[WiringStage] = []

    for block_idx in range(num_blocks):
        for layer_idx in range(block_idx + 1):
            m = 2 ** (block_idx - layer_idx)
            pairs: List[Tuple[int, int]] = []
            for i in range(0, m_pow, 2 * m):
                for j in range(m):
                    ix = i + j
                    a, b = ix, ix + m
                    if a >= offset and b >= offset:
                        la, lb = a - offset, b - offset
                        if (ix // (2 ** (block_idx + 1))) % 2 == 1:
                            la, lb = lb, la
                        pairs.append((la, lb))
            if pairs:
                stages.append(_pairs_to_stage_matrices(n, pairs))
    return stages


def odd_even_merge_network(n: int) -> List[WiringStage]:
    """Batcher's Odd-Even Mergesort network (Batcher 1968; Knuth TAOCP Vol. 3 Sec. 5.2.2).

    Achieves the same ``p*(p+1)/2`` parallel depth as Bitonic sort with strictly fewer
    comparators for ``n >= 4``. Handles arbitrary ``n >= 2`` by pruning wires ``>= n``.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    p_pow = 2 ** math.ceil(math.log2(n))
    stages: List[WiringStage] = []

    p = 1
    while p < p_pow:
        k = p
        while k >= 1:
            pairs: List[Tuple[int, int]] = []
            for j in range(k % p, p_pow - k, 2 * k):
                for i in range(min(k, p_pow - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        a, b = i + j, i + j + k
                        if a < n and b < n:
                            pairs.append((a, b))
            if pairs:
                stages.append(_pairs_to_stage_matrices(n, pairs))
            k //= 2
        p *= 2
    return stages


def odd_even_transposition_network(n: int) -> List[WiringStage]:
    """Habermann's odd-even transposition sorting network (depth ``n``)."""
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    stages: List[WiringStage] = []
    for layer in range(n):
        start = layer % 2
        pairs = [(i, i + 1) for i in range(start, n - 1, 2)]
        stages.append(_pairs_to_stage_matrices(n, pairs))
    return stages


def random_fixed_network(n: int, depth: int | None = None, seed: int = 42) -> List[WiringStage]:
    """Negative structural control: same parallel depth as Bitonic, random pairwise swaps."""
    if depth is None:
        depth = len(bitonic_network(n))
    rng = np.random.default_rng(seed)
    stages: List[WiringStage] = []
    for _ in range(depth):
        perm = rng.permutation(n)
        pairs = [(int(perm[2 * k]), int(perm[2 * k + 1])) for k in range(n // 2)]
        stages.append(_pairs_to_stage_matrices(n, pairs))
    return stages


def get_sorting_network(
    wiring_type: str,
    n: int,
    seed: int = 42,
) -> List[WiringStage]:
    """Factory returning a list of numpy ``(split_a, split_b, combine_min, combine_max)`` stages."""
    if wiring_type == "bitonic":
        return bitonic_network(n)
    if wiring_type == "odd_even_merge":
        return odd_even_merge_network(n)
    if wiring_type == "odd_even_trans":
        return odd_even_transposition_network(n)
    if wiring_type == "random_fixed":
        return random_fixed_network(n, seed=seed)
    raise ValueError(f"Unknown wiring_type: {wiring_type}")


def verify_exact_sorting_network(stages: List[WiringStage], n: int, trials: int = 256) -> bool:
    """Verify via hard min/max execution (and exhaustive 0-1 check for n <= 12) that wiring sorts."""
    if n <= 12:
        # Knuth's Zero-One Principle: sorts all real inputs iff it sorts all {0,1}^n vectors.
        grid = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(np.float64)
        x = grid
    else:
        rng = np.random.default_rng(12345)
        x = rng.uniform(-1.0, 1.0, size=(trials, n))
    expected = np.sort(x, axis=-1)
    cur = x.copy()
    for split_a, split_b, combine_min, combine_max in stages:
        a = cur @ split_a.T
        b = cur @ split_b.T
        mn = np.minimum(a, b)
        mx = np.maximum(a, b)
        cur = mn @ combine_min.T + mx @ combine_max.T
    return bool(np.allclose(cur, expected, atol=1e-12))


def stages_to_buffers(
    stages: List[WiringStage],
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Convert numpy wiring stages to typed torch tensors."""
    return [
        tuple(torch.from_numpy(m).to(dtype=dtype, device=device) for m in stage)  # type: ignore[misc]
        for stage in stages
    ]
