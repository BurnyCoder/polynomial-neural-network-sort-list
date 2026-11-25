"""Mechanical PyTorch FX purity tests guaranteeing zero non-polynomial ops in pure PNNs."""

from __future__ import annotations

import pytest

from pnn_sort.audit import audit_pure_polynomial
from pnn_sort.models import build_model

PURE_POLY_MODELS = [
    ("polysort_shared", {}),
    ("polysort_shared_m2", {}),
    ("polysort_adaptive", {}),
    ("polysort_adaptive_m2", {}),
    ("polysort_custom", {"mode": "unconstrained_odd_cheby", "shared": False}),
    ("polysort_custom", {"mode": "general_cheby", "shared": False}),
    ("polysort_custom", {"mode": "raw_monomial", "shared": False}),
    ("polysort_learned_wire", {}),
    ("poly_pairwise_rank", {}),
    ("deepsets_poly", {}),
    ("pinet_ncp_skip", {}),
    ("poly_mlp", {}),
    ("analytical_cheon_f1", {}),
    ("analytical_cheon_f2", {}),
    ("analytical_cheon_fg", {}),
    ("analytical_remez", {}),
    ("identity_control", {}),
    ("linear_control", {}),
]

NON_POLY_MODELS = [
    ("diffsort_logistic", {}),
    ("diffsort_cauchy", {}),
    ("diffsort_reciprocal", {}),
    ("relu_mlp", {}),
    ("polysort_scale_norm", {}),
]


@pytest.mark.parametrize("model_name,kwargs", PURE_POLY_MODELS)
def test_pure_polynomial_models_pass_fx_audit(model_name: str, kwargs: dict) -> None:
    model = build_model(model_name, n=8, **kwargs)
    assert getattr(model, "is_pure_polynomial", False) is True
    res = audit_pure_polynomial(model)
    assert res.passed, f"{model_name} failed FX pure-polynomial audit: {res.violations}"


def test_zero_shot_length_views_pass_fx_audit() -> None:
    for name in ("polysort_shared", "poly_pairwise_rank", "analytical_cheon_f1"):
        base = build_model(name, n=8)
        for new_n in (4, 16, 32):
            view = base.for_length(new_n)
            res = audit_pure_polynomial(view)
            assert res.passed, f"{name}.for_length({new_n}) failed FX audit: {res.violations}"


@pytest.mark.parametrize("model_name,kwargs", NON_POLY_MODELS)
def test_non_polynomial_baselines_fail_fx_audit(model_name: str, kwargs: dict) -> None:
    model = build_model(model_name, n=8, **kwargs)
    assert getattr(model, "is_pure_polynomial", True) is False
    res = audit_pure_polynomial(model)
    assert not res.passed, f"Expected non-poly model {model_name} to fail pure-polynomial FX audit!"
