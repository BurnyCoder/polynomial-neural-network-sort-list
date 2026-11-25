"""Strictly pure-polynomial neural building blocks and analytical CKKS polynomials.

Ground-truth references:
- Cheon, Kim & Kim (ASIACRYPT 2020, IACR ePrint 2019/1234):
  ``f_m(u) = sum_{i=0}^m (1 / 4^i) * binom(2i, i) * u * (1 - u^2)^i``.
- Lee, Lee, No & Kim (IEEE TDSC 2022, IACR ePrint 2020/834):
  Composite Remez minimax odd polynomials approximating sign(u) on [delta, 1].
- Chrysos et al. (IEEE TPAMI 2021, arXiv:2006.13026, Eq. 9 & 10):
  NCP and NCP-Skip tensor-factorized polynomial blocks (strictly without BN/tanh).
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np
from scipy.optimize import linprog
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Pure-polynomial Chebyshev basis helpers (additions & multiplications only)
# ---------------------------------------------------------------------------


def chebyshev_even_basis(u: torch.Tensor, num_terms: int) -> List[torch.Tensor]:
    """Compute [T_0(u), T_2(u), ..., T_{2*(m-1)}(u)] strictly via polynomial recurrence.

    Uses the exact trigonometric identity ``T_{2j}(u) = T_j(2u^2 - 1)``, cutting
    multiplicative steps in half while remaining 100% pure polynomial.
    """
    if num_terms <= 0:
        return []
    t0 = torch.ones_like(u)
    if num_terms == 1:
        return [t0]
    v = 2.0 * (u * u) - 1.0  # T_2(u)
    basis = [t0, v]
    for _ in range(2, num_terms):
        t_next = 2.0 * v * basis[-1] - basis[-2]
        basis.append(t_next)
    return basis


def chebyshev_odd_basis(u: torch.Tensor, num_terms: int) -> List[torch.Tensor]:
    """Compute [T_1(u), T_3(u), ..., T_{2m-1}(u)] strictly via polynomial recurrence."""
    if num_terms <= 0:
        return []
    if num_terms == 1:
        return [u]
    v = 2.0 * (u * u) - 1.0  # T_2(u)
    t3 = 2.0 * v * u - u  # 4u^3 - 3u = T_3(u)
    basis = [u, t3]
    for _ in range(2, num_terms):
        # T_{2j+1}(u) = 2*T_2(u)*T_{2j-1}(u) - T_{2j-3}(u)
        t_next = 2.0 * v * basis[-1] - basis[-2]
        basis.append(t_next)
    return basis


def chebyshev_all_basis(u: torch.Tensor, max_deg: int) -> List[torch.Tensor]:
    """Compute [T_0(u), T_1(u), ..., T_{max_deg}(u)] via exact 3-term recurrence."""
    t0 = torch.ones_like(u)
    if max_deg == 0:
        return [t0]
    basis = [t0, u]
    for _ in range(2, max_deg + 1):
        basis.append(2.0 * u * basis[-1] - basis[-2])
    return basis


# ---------------------------------------------------------------------------
# Analytical Cheon (2020) and Remez Minimax (Lee et al., 2022) coefficients
# ---------------------------------------------------------------------------


def cheon_fp_cheby_coeffs(num_terms: int) -> List[float]:
    """Exact Chebyshev coefficients c_0..c_{m-1} such that
    ``u + u(1 - u^2) * sum_{j=0}^{m-1} c_j T_{2j}(u) == f_m(u)`` identically,
    where ``f_m(u) = sum_{i=0}^m (1/4^i) * comb(2i, i) * u * (1 - u^2)^i`` (Cheon et al., 2020).
    """
    # Represent q(v) = sum_{i=1}^m (1/4^i) comb(2i, i) * ((1 - v)/2)^{i-1} in T_j(v)
    # where v = 2u^2 - 1 so 1 - u^2 = (1 - v) / 2 and T_j(v) = T_{2j}(u).
    # Build polynomial in power basis of v first, then project to Chebyshev T_j(v).
    poly_v = np.zeros(num_terms, dtype=np.float64)
    # (1 - v)/2 powers in standard monomial basis of v
    half_minus = np.array([0.5, -0.5], dtype=np.float64)
    cur_pow = np.array([1.0], dtype=np.float64)
    for i in range(1, num_terms + 1):
        coeff = math.comb(2 * i, i) / (4.0**i)
        poly_v[: len(cur_pow)] += coeff * cur_pow
        cur_pow = np.convolve(cur_pow, half_minus)
    cheb_c = np.polynomial.chebyshev.poly2cheb(poly_v)
    out = np.zeros(num_terms, dtype=np.float64)
    out[: len(cheb_c)] = cheb_c
    return [float(x) for x in out]


def _solve_odd_minimax_stage_lp(
    alpha: float,
    num_terms: int,
    grid_points: int = 400,
) -> tuple[np.ndarray, float, float, float]:
    """Solve exact global L_inf odd polynomial minimax on [alpha, 1] via HiGHS LP.

    Finds ``a_0..a_m`` minimizing ``tau = max_{u in [alpha, 1]} |1 - sum_{j=0}^m a_j T_{2j+1}(u)|``.
    Returns ``(coeffs, peak_on_unit, min_on_alpha_1, tau)``.
    """
    u = np.concatenate([
        np.geomspace(alpha, max(alpha * 2.0, 0.4), grid_points // 2, endpoint=False),
        np.linspace(max(alpha * 2.0, 0.4), 1.0, grid_points // 2),
    ])
    m = num_terms + 1
    basis = np.zeros((len(u), m), dtype=np.float64)
    for j in range(m):
        c = np.zeros(2 * j + 2, dtype=np.float64)
        c[2 * j + 1] = 1.0
        basis[:, j] = np.polynomial.chebyshev.chebval(u, c)

    c_obj = np.zeros(m + 1, dtype=np.float64)
    c_obj[-1] = 1.0
    A_ub = np.vstack([
        np.hstack([basis, -np.ones((len(u), 1))]),
        np.hstack([-basis, -np.ones((len(u), 1))]),
    ])
    b_ub = np.concatenate([np.ones(len(u)), -np.ones(len(u))])
    bounds = [(None, None)] * m + [(0.0, None)]
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"HiGHS minimax LP failed at alpha={alpha}: {res.message}")

    a = res.x[:m]
    tau = float(res.x[-1])
    u_full = np.linspace(0.0, 1.0, 2000)
    b_full = np.column_stack(
        [np.polynomial.chebyshev.chebval(u_full, np.eye(1, 2 * j + 2, 2 * j + 1)[0]) for j in range(m)]
    )
    peak = float(np.max(np.abs(b_full @ a)))
    lo = float(np.min(basis @ a))
    return a, peak, lo, tau


def fit_remez_odd_composite_coeffs(
    num_layers: int = 3,
    num_terms: int = 1,
    delta: float = 0.10,
    grid_points: int = 400,
) -> List[List[float]]:
    """Compute exact stage-by-stage minimax odd composite Chebyshev coefficients (Lee et al., IEEE TDSC 2022).

    Each stage ``k = 1..L-1`` solves the exact odd L_inf minimax problem on ``[alpha_{k-1}, 1]``
    via convex LP and rescales by ``1 / peak`` so its image lies inside ``[alpha_k, 1]`` (with
    ``alpha_k = (1 - tau_k) / (1 + tau_k)``). The final stage ``L`` is centered at ``1``
    (equioscillating in ``[1 - tau_L, 1 + tau_L]``), matching Lee et al. (2022) Algorithm 2.
    Output shape: ``(num_layers, num_terms + 1)`` for ``CompositePolySign(mode='unconstrained_odd_cheby')``.
    """
    alpha = float(delta)
    stages: List[List[float]] = []
    for k in range(num_layers):
        a, peak, lo, _tau = _solve_odd_minimax_stage_lp(alpha, num_terms, grid_points=grid_points)
        if k < num_layers - 1:
            stages.append((a / peak).tolist())
            alpha = max(1e-5, lo / peak)
        else:
            stages.append(a.tolist())
    return stages


# ---------------------------------------------------------------------------
# Learnable Composite Polynomial Sign Module (100% Pure Polynomial)
# ---------------------------------------------------------------------------


class CompositePolySign(nn.Module):
    """Strictly pure-polynomial composite approximator for ``sign(u)`` on ``[-1, 1]``.

    Parameterization modes (Ablation A1):
    - ``fixed_point_cheby`` (default):
        ``h_k(u) = u + u(1 - u^2) * sum_{j=0}^{m-1} c_{k,j} T_{2j}(u)``
        Degree per layer: ``2m + 1``. Guaranteed ``h_k(0)=0, h_k(+-1)=+-1``, odd symmetry.
    - ``unconstrained_odd_cheby``:
        ``h_k(u) = sum_{j=0}^{m} c_{k,j} T_{2j+1}(u)``
        Degree per layer: ``2m + 1``. Odd symmetry only (boundary free).
    - ``general_cheby``:
        ``h_k(u) = sum_{j=0}^{2m+1} c_{k,j} T_j(u)``
        Degree per layer: ``2m + 1``. Neither oddness nor fixed points enforced.
    - ``raw_monomial``:
        ``h_k(u) = sum_{j=0}^{m} c_{k,j} u^{2j+1}``
        Degree per layer: ``2m + 1``. Raw monomial basis (tests conditioning).
    """

    def __init__(
        self,
        num_layers: int = 3,
        num_terms: int = 1,
        mode: str = "fixed_point_cheby",
        init: str = "cheon",
        trainable: bool = True,
        init_coeffs: Sequence[Sequence[float]] | None = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_terms = num_terms
        self.mode = mode
        self.init_mode = init
        self.block_degree = 2 * num_terms + 1
        self.total_degree = self.block_degree**num_layers

        if mode == "fixed_point_cheby":
            shape = (num_layers, num_terms)
        elif mode in ("unconstrained_odd_cheby", "raw_monomial"):
            shape = (num_layers, num_terms + 1)
        elif mode == "general_cheby":
            shape = (num_layers, self.block_degree + 1)
        else:
            raise ValueError(f"Unknown CompositePolySign mode: {mode}")

        data = torch.zeros(shape, dtype=torch.float32)
        if init_coeffs is not None:
            data = torch.tensor(init_coeffs, dtype=torch.float32)
            assert tuple(data.shape) == shape, f"Expected shape {shape}, got {tuple(data.shape)}"
        else:
            self._fill_init(data)

        if trainable:
            self.coeffs = nn.Parameter(data)
        else:
            self.register_buffer("coeffs", data)

    def _fill_init(self, data: torch.Tensor) -> None:
        if self.mode == "fixed_point_cheby":
            if self.init_mode == "identity":
                data.zero_()
            elif self.init_mode == "cheon":
                c = torch.tensor(cheon_fp_cheby_coeffs(self.num_terms), dtype=data.dtype)
                data.copy_(c.unsqueeze(0).expand_as(data))
            elif self.init_mode == "random":
                nn.init.normal_(data, mean=0.25, std=0.15)
            else:
                raise ValueError(f"Unknown init: {self.init_mode}")
        elif self.mode == "unconstrained_odd_cheby":
            # T_1(u) = u gives identity; Cheon f_1 = 3/2 u - 1/2 u^3 = 9/8 T_1 - 1/8 T_3
            data.zero_()
            if self.init_mode == "identity":
                data[:, 0] = 1.0
            elif self.init_mode == "cheon":
                # Convert exact Cheon f_m to odd Chebyshev expansion
                poly = self._cheon_odd_monomials()
                cheb = np.polynomial.chebyshev.poly2cheb(poly)
                odd_cheb = cheb[1::2]
                data[:, : len(odd_cheb)] = torch.tensor(odd_cheb, dtype=data.dtype)
            elif self.init_mode == "random":
                data[:, 0] = 1.0
                data[:, 1:] = torch.randn_like(data[:, 1:]) * 0.1
        elif self.mode == "raw_monomial":
            data.zero_()
            if self.init_mode == "identity":
                data[:, 0] = 1.0
            elif self.init_mode == "cheon":
                poly = self._cheon_odd_monomials()
                odd_mono = poly[1::2]
                data[:, : len(odd_mono)] = torch.tensor(odd_mono, dtype=data.dtype)
            elif self.init_mode == "random":
                data[:, 0] = 1.0
                data[:, 1:] = torch.randn_like(data[:, 1:]) * 0.1
        elif self.mode == "general_cheby":
            data.zero_()
            if self.init_mode == "identity":
                data[:, 1] = 1.0
            elif self.init_mode == "cheon":
                poly = self._cheon_odd_monomials()
                cheb = np.polynomial.chebyshev.poly2cheb(poly)
                data[:, : len(cheb)] = torch.tensor(cheb, dtype=data.dtype)
            elif self.init_mode == "random":
                data[:, 1] = 1.0
                data[:, 2:] = torch.randn_like(data[:, 2:]) * 0.05

    def _cheon_odd_monomials(self) -> np.ndarray:
        """Exact monomial coefficients of Cheon f_m(u) = sum_{i=0}^m (1/4^i) comb(2i,i) u(1-u^2)^i."""
        m = self.num_terms
        deg = 2 * m + 1
        poly = np.zeros(deg + 1, dtype=np.float64)
        base = np.array([1.0, 0.0, -1.0], dtype=np.float64)  # 1 - u^2
        cur = np.array([0.0, 1.0], dtype=np.float64)  # u
        for i in range(m + 1):
            coeff = math.comb(2 * i, i) / (4.0**i)
            poly[: len(cur)] += coeff * cur
            if i < m:
                cur = np.convolve(cur, base)
        return poly

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        z = u
        for k in range(self.num_layers):
            ck = self.coeffs[k]
            if self.mode == "fixed_point_cheby":
                basis = chebyshev_even_basis(z, self.num_terms)
                q = ck[0] * basis[0]
                for j in range(1, self.num_terms):
                    q = q + ck[j] * basis[j]
                z = z + z * (1.0 - z * z) * q
            elif self.mode == "unconstrained_odd_cheby":
                basis = chebyshev_odd_basis(z, self.num_terms + 1)
                out = ck[0] * basis[0]
                for j in range(1, self.num_terms + 1):
                    out = out + ck[j] * basis[j]
                z = out
            elif self.mode == "raw_monomial":
                z2 = z * z
                p = z
                out = ck[0] * p
                for j in range(1, self.num_terms + 1):
                    p = p * z2
                    out = out + ck[j] * p
                z = out
            elif self.mode == "general_cheby":
                basis = chebyshev_all_basis(z, self.block_degree)
                out = ck[0] * basis[0]
                for j in range(1, self.block_degree + 1):
                    out = out + ck[j] * basis[j]
                z = out
        return z


# ---------------------------------------------------------------------------
# Chrysos et al. (TPAMI 2021) Pure Pi-Net NCP-Skip Block (Eq. 10, zero BN/tanh)
# ---------------------------------------------------------------------------


class PureNCPSkipBlock(nn.Module):
    """Strict pure-polynomial NCP-Skip block (Chrysos et al., arXiv:2006.13026, Eq. 10).

    Recursion for order N:
      x_1 = (A_1 z) * b_1
      x_k = (A_k z) * (S_k x_{k-1} + b_k) + V_k x_{k-1},  k = 2..N
      out = C x_N + beta
    Zero normalization or activation functions are applied, preserving strict polynomiality.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, order: int = 3) -> None:
        super().__init__()
        self.order = order
        self.a_layers = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(order)])
        self.s_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(order - 1)])
        self.v_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(order - 1)])
        self.b1 = nn.Parameter(torch.ones(hidden_dim))
        self.out_proj = nn.Linear(hidden_dim, out_dim, bias=True)
        self._init_stable()

    def _init_stable(self) -> None:
        # Small-variance init to prevent high-order polynomial explosion at step 0
        for lin in self.a_layers:
            nn.init.normal_(lin.weight, mean=0.0, std=0.25 / math.sqrt(lin.in_features))
        for lin in self.s_layers:
            nn.init.normal_(lin.weight, mean=0.0, std=0.25 / math.sqrt(lin.in_features))
            nn.init.constant_(lin.bias, 0.5)
        for lin in self.v_layers:
            nn.init.eye_(lin.weight)
            lin.weight.data.mul_(0.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.a_layers[0](z) * self.b1
        for k in range(1, self.order):
            x = self.a_layers[k](z) * self.s_layers[k - 1](x) + self.v_layers[k - 1](x)
        return self.out_proj(x)
