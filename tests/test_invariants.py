"""Algebraic invariant, symbolic math, wiring correctness, and Mazzone tie-correction tests."""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp
import torch

from pnn_sort.models import (
    CompositePolySign,
    PolyPairwiseRank,
    PolySortNet,
    build_analytical_cheon_fg_sort,
    build_analytical_cheon_sort,
    build_analytical_remez_sort,
    cheon_fp_cheby_coeffs,
    get_sorting_network,
    verify_exact_sorting_network,
)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8, 12, 16, 32])
@pytest.mark.parametrize("wiring_type", ["bitonic", "odd_even_merge", "odd_even_trans"])
def test_sorting_network_wiring_exact(n: int, wiring_type: str) -> None:
    stages = get_sorting_network(wiring_type, n)
    assert verify_exact_sorting_network(stages, n), f"{wiring_type} failed exact sorting verification at n={n}"


def test_cheon_chebyshev_expansion_matches_symbolic_exact() -> None:
    u = sp.Symbol("u")
    for m in (1, 2, 3, 4):
        # Exact Cheon et al. (ASIACRYPT 2020) formula
        f_exact = sp.expand(sum((sp.Rational(math.comb(2 * i, i), 4**i)) * u * (1 - u**2) ** i for i in range(m + 1)))
        coeffs = cheon_fp_cheby_coeffs(m)
        # Reconstruct u + u(1 - u^2) * sum c_j T_{2j}(u)
        f_cheb = u + u * (1 - u**2) * sum(coeffs[j] * sp.chebyshevt(2 * j, u) for j in range(m))
        f_cheb = sp.expand(f_cheb)
        diff_poly = sp.Poly(f_exact - f_cheb, u)
        max_coeff_err = max(abs(float(c)) for c in diff_poly.all_coeffs()) if diff_poly.all_coeffs() else 0.0
        assert max_coeff_err < 1e-13, f"Cheon f_{m} mismatch: {max_coeff_err}"


def test_polysort_exact_algebraic_invariants_float64() -> None:
    torch.manual_seed(0)
    model = PolySortNet(n=8, wiring_type="bitonic", num_layers=3, num_terms=2, shared=True, init="random").double()
    x = (torch.rand(64, 8, dtype=torch.float64) * 2.0) - 1.0
    y = model(x)

    # 1. Exact sum conservation: sum(y) == sum(x)
    sum_err = torch.max(torch.abs(y.sum(dim=-1) - x.sum(dim=-1))).item()
    assert sum_err < 1e-13, f"Sum conservation violated: {sum_err}"

    # 2. Exact arbitrary shift equivariance: f(x + c*1) == f(x) + c*1
    for c in (1.0, 10.0, 100.0):
        y_shift = model(x + c)
        shift_err = torch.max(torch.abs((y_shift - c) - y)).item()
        assert shift_err < 1e-11, f"Shift equivariance violated for c={c}: {shift_err}"

    # 3. Exact negation anti-symmetry: f(-x) == -rev(f(x))
    y_neg = model(-x)
    neg_err = torch.max(torch.abs(y_neg - (-torch.flip(y, dims=[-1])))).item()
    assert neg_err < 1e-13, f"Negation anti-symmetry violated: {neg_err}"

    # 4. Exact idempotency on all-equal constant vectors
    x_const = torch.full((8, 8), 0.37, dtype=torch.float64)
    const_err = torch.max(torch.abs(model(x_const) - x_const)).item()
    assert const_err < 1e-14, f"Constant idempotency violated: {const_err}"


def test_analytical_cheon_and_remez_sort_accuracy_and_boundedness() -> None:
    cheon_l3 = build_analytical_cheon_sort(n=8, num_layers=3, num_terms=1).double()
    cheon_fg = build_analytical_cheon_fg_sort(n=8).double()
    remez = build_analytical_remez_sort(n=8, num_layers=3, num_terms=1, delta=0.10).double()

    torch.manual_seed(7)
    x = torch.rand(500, 8, dtype=torch.float64) * 2.0 - 1.0
    y = torch.sort(x, dim=-1).values

    # Cheon f1^(3) has origin slope 1.5^3 = 3.375 -> analytical MAE ~0.11; f3 o g3 & Remez are much sharper!
    p_l3 = cheon_l3(x)
    p_fg = cheon_fg(x)
    p_rm = remez(x)
    mae_l3 = float((p_l3 - y).abs().mean().item())
    mae_fg = float((p_fg - y).abs().mean().item())
    mae_rm = float((p_rm - y).abs().mean().item())
    assert mae_l3 < 0.13, f"cheon f1^(3) MAE unexpectedly high: {mae_l3}"
    assert mae_fg < 0.03, f"cheon f3 o g3 MAE unexpectedly high: {mae_fg}"
    assert mae_rm < 0.055 and mae_rm < mae_l3, f"Expected Remez L=3 MAE ({mae_rm}) < 0.055 < Cheon ({mae_l3})"
    for p in (p_l3, p_fg):
        assert float(p.min().item()) >= -1.01 and float(p.max().item()) <= 1.01
    assert torch.isfinite(p_rm).all() and float(p_rm.abs().max().item()) <= 1.5

    # Remez minimax sign L_inf error on [0.10, 1.0] must strictly beat Cheon f_1^(3)
    u = torch.linspace(0.10, 1.0, 500, dtype=torch.float64)
    err_c = float((1.0 - cheon_l3.sign_shared(u)).abs().max().item())
    err_r = float((1.0 - remez.sign_shared(u)).abs().max().item())
    assert err_r < err_c, f"Expected Remez L_inf ({err_r}) < Cheon L_inf ({err_c})"


def test_mazzone_polynomial_tie_correction_exact() -> None:
    # With high-depth analytical sign, PolyPairwiseRank should handle exact duplicates cleanly!
    model = PolyPairwiseRank(n=4, cmp_layers=5, cmp_terms=2, ind_layers=5, ind_terms=2, init="cheon").double()
    x = torch.tensor([[0.4, -0.2, -0.2, 0.9], [-0.5, -0.5, -0.5, 0.5]], dtype=torch.float64)
    y_true = torch.sort(x, dim=-1).values
    y_pred = model(x)
    err = float((y_pred - y_true).abs().max().item())
    assert err < 1e-3, f"PolyPairwiseRank tie-correction error too large: {err}, pred={y_pred}"
