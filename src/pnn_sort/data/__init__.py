"""Deterministic, leak-free synthetic list datasets and OOD stress suites."""

from .datasets import (
    dataset_row_hashes,
    generate_ood_suites,
    generate_split,
    make_rng,
    verify_disjoint_splits,
)

__all__ = [
    "make_rng",
    "generate_split",
    "generate_ood_suites",
    "dataset_row_hashes",
    "verify_disjoint_splits",
]
