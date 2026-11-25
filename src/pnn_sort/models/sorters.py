"""Unified model zoo: strictly pure PNNs, analytical CKKS sorters, controls, and non-poly baselines.

Every model exposes standardized metadata:
- ``is_pure_polynomial: bool`` (verified mechanically by ``audit_pure_polynomial``)
- ``supports_variable_length: bool`` (supports ``model.for_length(new_n)``)
- ``effective_degree: int | None``
- ``seq_mult_depth: int | None``
"""

from __future__ import annotations

import math
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .poly_blocks import (
    CompositePolySign,
    PureNCPSkipBlock,
    chebyshev_all_basis,
    cheon_fp_cheby_coeffs,
    fit_remez_odd_composite_coeffs,
)
from .wiring import get_sorting_network, stages_to_buffers


def _register_wiring_buffers(
    module: nn.Module,
    wiring_type: str,
    n: int,
    wiring_seed: int = 42,
) -> int:
    stages = get_sorting_network(wiring_type, n, seed=wiring_seed)
    t_stages = stages_to_buffers(stages, dtype=torch.float32)
    for idx, (sa, sb, cmin, cmax) in enumerate(t_stages):
        module.register_buffer(f"split_a_{idx}", sa)
        module.register_buffer(f"split_b_{idx}", sb)
        module.register_buffer(f"combine_min_{idx}", cmin)
        module.register_buffer(f"combine_max_{idx}", cmax)
    return len(t_stages)


def _get_wiring_stage(
    module: nn.Module, idx: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        getattr(module, f"split_a_{idx}"),
        getattr(module, f"split_b_{idx}"),
        getattr(module, f"combine_min_{idx}"),
        getattr(module, f"combine_max_{idx}"),
    )


# ---------------------------------------------------------------------------
# 1. Main Method: PolySortNet (Shared & Stage-Adaptive Composite Polynomial Sorting Net)
# ---------------------------------------------------------------------------


class PolySortNet(nn.Module):
    """Strictly pure Polynomial Sorting Network (Main Method).

    Replaces discrete compare-and-swap (CAS) gates in a Bitonic / Odd-Even sorting
    network with learned odd composite Chebyshev polynomials:
      Delta = (b - a) * 0.5
      abs_half = Delta * s_theta(Delta)
      min_out = (a + b) * 0.5 - abs_half
      max_out = (a + b) * 0.5 + abs_half
    """

    def __init__(
        self,
        n: int,
        wiring_type: str = "bitonic",
        num_layers: int = 3,
        num_terms: int = 1,
        shared: bool = True,
        mode: str = "fixed_point_cheby",
        init: str = "cheon",
        trainable: bool = True,
        wiring_seed: int = 42,
        init_coeffs: List[List[float]] | None = None,
        _shared_sign_module: CompositePolySign | None = None,
    ) -> None:
        super().__init__()
        self.n = n
        self.wiring_type = wiring_type
        self.num_layers = num_layers
        self.num_terms = num_terms
        self.shared = shared
        self.mode = mode
        self.init_mode = init
        self.trainable = trainable
        self.wiring_seed = wiring_seed

        self.num_stages = _register_wiring_buffers(self, wiring_type, n, wiring_seed=wiring_seed)
        if shared:
            if _shared_sign_module is not None:
                self.sign_shared = _shared_sign_module
            else:
                self.sign_shared = CompositePolySign(
                    num_layers=num_layers,
                    num_terms=num_terms,
                    mode=mode,
                    init=init,
                    trainable=trainable,
                    init_coeffs=init_coeffs,
                )
            self.sign_stages = None
        else:
            self.sign_shared = None
            self.sign_stages = nn.ModuleList(
                [
                    CompositePolySign(
                        num_layers=num_layers,
                        num_terms=num_terms,
                        mode=mode,
                        init=init,
                        trainable=trainable,
                        init_coeffs=init_coeffs,
                    )
                    for _ in range(self.num_stages)
                ]
            )

        self.is_pure_polynomial = True
        self.supports_variable_length = shared and wiring_type != "random_fixed"
        comp_deg = (2 * num_terms + 1) ** num_layers
        self.comparator_degree = comp_deg
        # Total polynomial degree upper bound across stages: (comp_deg + 1)^num_stages
        self.effective_degree = (comp_deg + 1) ** self.num_stages
        self.seq_mult_depth = self.num_stages * num_layers

    def for_length(self, new_n: int) -> "PolySortNet":
        """Create a zero-shot view sharing the exact trained comparator for list length ``new_n``."""
        if not self.supports_variable_length:
            raise ValueError("Only shared-gate sorting networks support zero-shot length generalization.")
        ref = next(self.buffers())
        clone = PolySortNet(
            n=new_n,
            wiring_type=self.wiring_type,
            num_layers=self.num_layers,
            num_terms=self.num_terms,
            shared=True,
            mode=self.mode,
            init=self.init_mode,
            trainable=self.trainable,
            wiring_seed=self.wiring_seed,
            _shared_sign_module=self.sign_shared,
        )
        return clone.to(device=ref.device, dtype=ref.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cur = x
        for idx in range(self.num_stages):
            sa, sb, cmin, cmax = _get_wiring_stage(self, idx)
            a = F.linear(cur, sa)
            b = F.linear(cur, sb)
            half_sum = (a + b) * 0.5
            delta = (b - a) * 0.5
            sign_fn = self.sign_shared if self.sign_shared is not None else self.sign_stages[idx]
            abs_half = delta * sign_fn(delta)
            mn = half_sum - abs_half
            mx = half_sum + abs_half
            cur = F.linear(mn, cmin) + F.linear(mx, cmax)
        return cur


# ---------------------------------------------------------------------------
# 2. Learned-Wiring Polynomial Sorting Network (Pure Polynomial)
# ---------------------------------------------------------------------------


class PolySortNetLearnedWire(nn.Module):
    """Pure-polynomial sorting network with learned linear routing between adjacent CAS gates.

    Tests whether gradient descent can learn sorting wiring topology from random orthogonal
    initializations without being given Batcher's fixed permutations.
    """

    def __init__(
        self,
        n: int,
        num_stages: int | None = None,
        num_layers: int = 3,
        num_terms: int = 1,
        init: str = "cheon",
    ) -> None:
        super().__init__()
        if n % 2 != 0:
            raise ValueError("PolySortNetLearnedWire requires even n.")
        self.n = n
        self.num_stages = num_stages if num_stages is not None else len(get_sorting_network("bitonic", n))
        self.num_layers = num_layers
        self.num_terms = num_terms
        half = n // 2

        # Fixed adjacent pair selectors/combiners; all cross-wire routing is learned by R_t.
        sa = np.zeros((half, n), dtype=np.float32)
        sb = np.zeros((half, n), dtype=np.float32)
        cmin = np.zeros((n, half), dtype=np.float32)
        cmax = np.zeros((n, half), dtype=np.float32)
        for k in range(half):
            sa[k, 2 * k] = 1.0
            sb[k, 2 * k + 1] = 1.0
            cmin[2 * k, k] = 1.0
            cmax[2 * k + 1, k] = 1.0
        self.register_buffer("pair_sa", torch.from_numpy(sa))
        self.register_buffer("pair_sb", torch.from_numpy(sb))
        self.register_buffer("pair_cmin", torch.from_numpy(cmin))
        self.register_buffer("pair_cmax", torch.from_numpy(cmax))

        self.routers = nn.ModuleList([nn.Linear(n, n, bias=False) for _ in range(self.num_stages)])
        for lin in self.routers:
            nn.init.orthogonal_(lin.weight)
        self.out_router = nn.Linear(n, n, bias=False)
        nn.init.orthogonal_(self.out_router.weight)

        self.sign_stages = nn.ModuleList(
            [
                CompositePolySign(num_layers=num_layers, num_terms=num_terms, mode="fixed_point_cheby", init=init)
                for _ in range(self.num_stages)
            ]
        )

        self.is_pure_polynomial = True
        self.supports_variable_length = False
        comp_deg = (2 * num_terms + 1) ** num_layers
        self.effective_degree = (comp_deg + 1) ** self.num_stages
        self.seq_mult_depth = self.num_stages * num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cur = x
        for idx in range(self.num_stages):
            z = self.routers[idx](cur)
            a = F.linear(z, self.pair_sa)
            b = F.linear(z, self.pair_sb)
            half_sum = (a + b) * 0.5
            delta = (b - a) * 0.5
            abs_half = delta * self.sign_stages[idx](delta)
            mn = half_sum - abs_half
            mx = half_sum + abs_half
            cur = F.linear(mn, self.pair_cmin) + F.linear(mx, self.pair_cmax)
        return self.out_router(cur)


# ---------------------------------------------------------------------------
# 3. Constant-Depth All-to-All Polynomial Ranker (Mazzone et al., USENIX '25)
# ---------------------------------------------------------------------------


class PolyPairwiseRank(nn.Module):
    """Constant-comparison-depth all-to-all pure-polynomial sorter (Mazzone et al., arXiv:2412.15126).

    1. Pairwise comparisons: ``C[i, j] = (1 + s_cmp((x_j - x_i) / 2)) / 2``
    2. Exact quadratic tie-correction (Mazzone Sec. 4, Alg. 6):
       ``E = 4 * C * (1 - C)``
       ``r_j = sum_i C[i, j] + sum_{i <= j} E[i, j] - 0.5 * sum_i E[i, j]``
    3. Parallel polynomial indicator around target ranks ``k = 1..n``:
       ``W[j, k] = 0.25 * (1 + s_ind((r_j - (k - 0.5))/n)) * (1 - s_ind((r_j - (k + 0.5))/n))``
    4. Extraction: ``y_k = sum_j W[j, k] * x_j``
    """

    def __init__(
        self,
        n: int,
        cmp_layers: int = 3,
        cmp_terms: int = 2,
        ind_layers: int = 3,
        ind_terms: int = 2,
        init: str = "cheon",
        _cmp_sign: CompositePolySign | None = None,
        _ind_sign: CompositePolySign | None = None,
    ) -> None:
        super().__init__()
        self.n = n
        self.inv_n = 1.0 / float(n)
        self.cmp_layers = cmp_layers
        self.cmp_terms = cmp_terms
        self.ind_layers = ind_layers
        self.ind_terms = ind_terms
        self.init_mode = init

        self.cmp_sign = (
            _cmp_sign
            if _cmp_sign is not None
            else CompositePolySign(num_layers=cmp_layers, num_terms=cmp_terms, mode="fixed_point_cheby", init=init)
        )
        self.ind_sign = (
            _ind_sign
            if _ind_sign is not None
            else CompositePolySign(num_layers=ind_layers, num_terms=ind_terms, mode="fixed_point_cheby", init=init)
        )

        # Upper-triangular mask M[i, j] = 1 iff i <= j (for Mazzone Alg. 6 tie offset)
        upper = torch.triu(torch.ones(n, n, dtype=torch.float32))
        k_vals = torch.arange(1, n + 1, dtype=torch.float32)
        self.register_buffer("upper_mask", upper)
        self.register_buffer("k_lo", k_vals - 0.5)
        self.register_buffer("k_hi", k_vals + 0.5)

        self.is_pure_polynomial = True
        self.supports_variable_length = True
        d_cmp = (2 * cmp_terms + 1) ** cmp_layers
        d_ind = (2 * ind_terms + 1) ** ind_layers
        self.effective_degree = 1 + 2 * d_ind * (2 * d_cmp)
        self.seq_mult_depth = cmp_layers + 1 + ind_layers + 1

    def for_length(self, new_n: int) -> "PolyPairwiseRank":
        ref = next(self.buffers())
        clone = PolyPairwiseRank(
            n=new_n,
            cmp_layers=self.cmp_layers,
            cmp_terms=self.cmp_terms,
            ind_layers=self.ind_layers,
            ind_terms=self.ind_terms,
            init=self.init_mode,
            _cmp_sign=self.cmp_sign,
            _ind_sign=self.ind_sign,
        )
        return clone.to(device=ref.device, dtype=ref.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # diff[:, i, j] = (x[:, j] - x[:, i]) * 0.5
        diff = (x.unsqueeze(-2) - x.unsqueeze(-1)) * 0.5
        c_mat = 0.5 + 0.5 * self.cmp_sign(diff)
        # Mazzone Sec. 4 exact quadratic equality indicator E = 4 * C * (1 - C)
        e_mat = 4.0 * c_mat * (1.0 - c_mat)
        u_vec = (e_mat * self.upper_mask).sum(dim=-2)
        t_vec = e_mat.sum(dim=-2)
        ranks = c_mat.sum(dim=-2) + u_vec - 0.5 * t_vec  # (B, n)

        z_lo = (ranks.unsqueeze(-1) - self.k_lo) * self.inv_n  # (B, n, n)
        z_hi = (ranks.unsqueeze(-1) - self.k_hi) * self.inv_n  # (B, n, n)
        w_mat = (0.5 + 0.5 * self.ind_sign(z_lo)) * (0.5 - 0.5 * self.ind_sign(z_hi))
        return torch.einsum("bj,bjk->bk", x, w_mat)


# ---------------------------------------------------------------------------
# 4. Symmetric Power-Sum / DeepSets Pure PNN (Zaheer 2017; Wagstaff 2022)
# ---------------------------------------------------------------------------


class DeepSetsPoly(nn.Module):
    """Permutation-invariant pure Polynomial Neural Network (DeepSets / Symmetric Polynomials).

    Computes elementwise Chebyshev moments ``[T_1(x_i), ..., T_K(x_i)]``, embeds into
    latent width ``latent_dim >= max(64, 4*n)`` (satisfying Wagstaff et al. 2019/2022
    Theorem on latent dimension >= n), averages symmetrically across set elements,
    and decodes order statistics via a pure ProdPoly NCP-Skip head.
    """

    def __init__(self, n: int, poly_deg: int = 6, latent_dim: int = 64, hidden_dim: int = 96) -> None:
        super().__init__()
        self.n = n
        self.poly_deg = poly_deg
        eff_latent = max(latent_dim, 4 * n)
        self.phi = nn.Linear(poly_deg, eff_latent, bias=True)
        self.rho1 = PureNCPSkipBlock(eff_latent, hidden_dim, hidden_dim, order=3)
        self.rho2 = PureNCPSkipBlock(hidden_dim, hidden_dim, n, order=3)
        nn.init.xavier_uniform_(self.phi.weight, gain=0.5)
        nn.init.zeros_(self.phi.bias)

        self.is_pure_polynomial = True
        self.supports_variable_length = False
        self.effective_degree = poly_deg * 3 * 3
        self.seq_mult_depth = 6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stack [T_1(x), ..., T_K(x)] -> (B, n, K)
        basis = chebyshev_all_basis(x, self.poly_deg)[1:]
        feats = torch.stack(basis, dim=-1)
        pooled = self.phi(feats).mean(dim=-2)  # exact permutation invariance
        h = self.rho1(pooled)
        return self.rho2(h)


# ---------------------------------------------------------------------------
# 5. Unstructured Pi-Net NCP-Skip (Chrysos et al., CVPR 2020 / TPAMI 2021)
# ---------------------------------------------------------------------------


class PiNetSorter(nn.Module):
    """Unstructured pure ProdPoly NCP-Skip (Chrysos et al., arXiv:2006.13026, Eq. 10 & Sec. 3.2)."""

    def __init__(self, n: int, hidden_dim: int = 128, order: int = 3, num_blocks: int = 2) -> None:
        super().__init__()
        self.n = n
        self.order = order
        self.num_blocks = num_blocks
        blocks: List[nn.Module] = []
        in_d = n
        for idx in range(num_blocks):
            out_d = n if idx == num_blocks - 1 else hidden_dim
            blocks.append(PureNCPSkipBlock(in_d, hidden_dim, out_d, order=order))
            in_d = out_d
        self.blocks = nn.ModuleList(blocks)

        self.is_pure_polynomial = True
        self.supports_variable_length = False
        self.effective_degree = order**num_blocks
        self.seq_mult_depth = num_blocks * order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for blk in self.blocks:
            h = blk(h)
        return h


# ---------------------------------------------------------------------------
# 6. Unstructured Polynomial-Activation MLP (Kileel et al., NeurIPS 2019)
# ---------------------------------------------------------------------------


class ChebyshevActivation(nn.Module):
    """Per-neuron learnable pure degree-3 Chebyshev polynomial activation (zero tanh clamping)."""

    def __init__(self, dim: int, deg: int = 3) -> None:
        super().__init__()
        self.deg = deg
        coeffs = torch.zeros(deg + 1, dim)
        coeffs[1] = 1.0  # initialize at exact identity T_1(z) = z
        self.coeffs = nn.Parameter(coeffs)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        basis = chebyshev_all_basis(z, self.deg)
        out = self.coeffs[0] * basis[0]
        for k in range(1, self.deg + 1):
            out = out + self.coeffs[k] * basis[k]
        return out


class PolyMLP(nn.Module):
    """Dense feedforward network with learnable degree-3 Chebyshev polynomial activations."""

    def __init__(self, n: int, hidden_dim: int = 128, depth: int = 3, act_deg: int = 3) -> None:
        super().__init__()
        self.n = n
        self.layers = nn.ModuleList()
        self.acts = nn.ModuleList()
        in_d = n
        for _ in range(depth):
            lin = nn.Linear(in_d, hidden_dim)
            nn.init.normal_(lin.weight, mean=0.0, std=0.4 / math.sqrt(in_d))
            nn.init.zeros_(lin.bias)
            self.layers.append(lin)
            self.acts.append(ChebyshevActivation(hidden_dim, deg=act_deg))
            in_d = hidden_dim
        self.out_proj = nn.Linear(hidden_dim, n)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        nn.init.zeros_(self.out_proj.bias)

        self.is_pure_polynomial = True
        self.supports_variable_length = False
        self.effective_degree = act_deg**depth
        self.seq_mult_depth = depth * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for lin, act in zip(self.layers, self.acts):
            h = act(lin(h))
        return self.out_proj(h)


# ---------------------------------------------------------------------------
# 7. Zero-Training Analytical CKKS & Trivial/Linear Controls
# ---------------------------------------------------------------------------


def build_analytical_cheon_sort(
    n: int,
    num_layers: int = 3,
    num_terms: int = 1,
    wiring_type: str = "bitonic",
) -> PolySortNet:
    """Zero-training analytical Cheon et al. (ASIACRYPT 2020) composite polynomial sorting network."""
    coeffs = [cheon_fp_cheby_coeffs(num_terms) for _ in range(num_layers)]
    return PolySortNet(
        n=n,
        wiring_type=wiring_type,
        num_layers=num_layers,
        num_terms=num_terms,
        shared=True,
        mode="fixed_point_cheby",
        trainable=False,
        init_coeffs=coeffs,
    )


def build_analytical_cheon_fg_sort(
    n: int,
    wiring_type: str = "bitonic",
) -> PolySortNet:
    """Zero-training Cheon et al. (2020) / Hong et al. (2021) mixed f_3 o g_3 sorting network.

    Exact polynomials verified from Mazzone et al. (USENIX Security 2025, arXiv:2412.15126, Sec. 2.2):
      g_3(x) = (4589x - 16577x^3 + 25614x^5 - 12860x^7) / 1024
      f_3(x) = (35x - 35x^3 + 21x^5 - 5x^7) / 16
    Degree per gate: 7 * 7 = 49 (multiplicative depth 2 * num_stages).
    """
    g3_mono = [4589.0 / 1024.0, -16577.0 / 1024.0, 25614.0 / 1024.0, -12860.0 / 1024.0]
    f3_mono = [35.0 / 16.0, -35.0 / 16.0, 21.0 / 16.0, -5.0 / 16.0]
    return PolySortNet(
        n=n,
        wiring_type=wiring_type,
        num_layers=2,
        num_terms=3,
        shared=True,
        mode="raw_monomial",
        trainable=False,
        init_coeffs=[g3_mono, f3_mono],
    )


def build_analytical_remez_sort(
    n: int,
    num_layers: int = 3,
    num_terms: int = 1,
    delta: float = 0.10,
    wiring_type: str = "bitonic",
) -> PolySortNet:
    """Zero-training Remez minimax composite polynomial sorting network (Lee et al., IEEE TDSC 2022)."""
    coeffs = fit_remez_odd_composite_coeffs(num_layers=num_layers, num_terms=num_terms, delta=delta)
    return PolySortNet(
        n=n,
        wiring_type=wiring_type,
        num_layers=num_layers,
        num_terms=num_terms,
        shared=True,
        mode="unconstrained_odd_cheby",
        trainable=False,
        init_coeffs=coeffs,
    )


class IdentityControl(nn.Module):
    """Trivial pass-through lower bound: ``y = x``."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n
        self.is_pure_polynomial = True
        self.supports_variable_length = True
        self.effective_degree = 1
        self.seq_mult_depth = 0

    def for_length(self, new_n: int) -> "IdentityControl":
        return IdentityControl(new_n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LinearControl(nn.Module):
    """Degree-1 affine control ``y = Wx + b`` (supports exact closed-form ridge/OLS fit)."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n
        self.linear = nn.Linear(n, n, bias=True)
        self.is_pure_polynomial = True
        self.supports_variable_length = False
        self.effective_degree = 1
        self.seq_mult_depth = 0

    @torch.no_grad()
    def fit_closed_form(self, x: torch.Tensor, y: torch.Tensor, ridge: float = 1e-6) -> None:
        xd = x.detach().double()
        yd = y.detach().double()
        ones = torch.ones((xd.shape[0], 1), dtype=torch.float64, device=xd.device)
        xa = torch.cat([xd, ones], dim=-1)
        xtx = xa.T @ xa + ridge * torch.eye(xa.shape[1], dtype=torch.float64, device=xd.device)
        xty = xa.T @ yd
        sol = torch.linalg.solve(xtx, xty)
        self.linear.weight.copy_(sol[:-1].T.to(self.linear.weight.dtype))
        self.linear.bias.copy_(sol[-1].to(self.linear.bias.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# 8. Non-Polynomial Baselines & Non-Polynomial Scale-Normalization Ablation
# ---------------------------------------------------------------------------


class DiffSortBaseline(nn.Module):
    """Differentiable Sorting Network baseline (Petersen et al., ICML 2021 & ICLR 2022).

    Implements the exact conditional swap relaxations verified in ``Felix-Petersen/diffsort``:
    - ``logistic``: ``sigmoid(beta * (b - a))``
    - ``cauchy``: ``(1 / pi) * atan(beta * (b - a)) + 0.5`` (monotonic, C^inf)
    - ``reciprocal``: ``0.5 * (v / (2 + |v|) + 1)`` (monotonic)
    Learns per-layer (or shared) log-steepness ``beta > 0``.
    """

    def __init__(
        self,
        n: int,
        distribution: str = "cauchy",
        wiring_type: str = "bitonic",
        init_steepness: float = 10.0,
        shared: bool = True,
        _shared_log_steepness: nn.Parameter | None = None,
    ) -> None:
        super().__init__()
        self.n = n
        self.distribution = distribution
        self.wiring_type = wiring_type
        self.shared = shared
        self.num_stages = _register_wiring_buffers(self, wiring_type, n)
        if shared:
            self.log_steepness = (
                _shared_log_steepness
                if _shared_log_steepness is not None
                else nn.Parameter(torch.tensor(math.log(init_steepness), dtype=torch.float32))
            )
        else:
            self.log_steepness = nn.Parameter(
                torch.full((self.num_stages,), math.log(init_steepness), dtype=torch.float32)
            )
        self.is_pure_polynomial = False
        self.supports_variable_length = shared
        self.effective_degree = None
        self.seq_mult_depth = self.num_stages

    def for_length(self, new_n: int) -> "DiffSortBaseline":
        if not self.supports_variable_length:
            raise ValueError("Only shared diffsort supports zero-shot length generalization.")
        ref = next(self.buffers())
        clone = DiffSortBaseline(
            n=new_n,
            distribution=self.distribution,
            wiring_type=self.wiring_type,
            shared=True,
            _shared_log_steepness=self.log_steepness,
        )
        return clone.to(device=ref.device, dtype=ref.dtype)

    def _alpha(self, diff: torch.Tensor, steepness: torch.Tensor) -> torch.Tensor:
        v = steepness * diff
        if self.distribution == "logistic":
            return torch.sigmoid(v)
        if self.distribution == "cauchy":
            return (1.0 / math.pi) * torch.atan(v) + 0.5
        if self.distribution == "reciprocal":
            return 0.5 * (v / (2.0 + v.abs()) + 1.0)
        raise ValueError(f"Unknown diffsort distribution: {self.distribution}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cur = x
        steep_all = torch.exp(torch.clamp(self.log_steepness, -4.0, 8.0))
        for idx in range(self.num_stages):
            sa, sb, cmin, cmax = _get_wiring_stage(self, idx)
            a = F.linear(cur, sa)
            b = F.linear(cur, sb)
            s = steep_all if self.shared else steep_all[idx]
            alpha = self._alpha(b - a, s)
            mn = alpha * a + (1.0 - alpha) * b
            mx = (1.0 - alpha) * a + alpha * b
            cur = F.linear(mn, cmin) + F.linear(mx, cmax)
        return cur


class ReLUMLPBaseline(nn.Module):
    """Parameter-matched non-polynomial GELU/ReLU MLP baseline."""

    def __init__(self, n: int, hidden_dim: int = 128, depth: int = 3) -> None:
        super().__init__()
        self.n = n
        layers: List[nn.Module] = []
        in_d = n
        for _ in range(depth):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.GELU())
            in_d = hidden_dim
        layers.append(nn.Linear(hidden_dim, n))
        self.net = nn.Sequential(*layers)
        self.is_pure_polynomial = False
        self.supports_variable_length = False
        self.effective_degree = None
        self.seq_mult_depth = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SetTransformerBaseline(nn.Module):
    """Permutation-invariant attention baseline (Lee et al., ICML 2019, arXiv:1810.00825).

    Encodes scalar set elements with 2 Self-Attention Blocks (SAB) and decodes sorted
    order statistics using multi-head attention from rank-quantile queries ``q_k in [0, 1]``
    (enabling zero-shot evaluation at unseen lengths ``new_n``).
    """

    def __init__(self, n: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.n = n
        self.d_model = d_model
        self.in_embed = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.query_mlp = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.out_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

        self.is_pure_polynomial = False
        self.supports_variable_length = True
        self.effective_degree = None
        self.seq_mult_depth = None

    def for_length(self, new_n: int) -> "SetTransformerBaseline":
        # Naturally supports arbitrary sequence length in forward()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = x.shape
        tokens = self.in_embed(x.unsqueeze(-1))
        enc = self.encoder(tokens)
        # Quantile coordinates in [0, 1] + centered [-1, 1] for length-agnostic slot queries
        q_pos = torch.linspace(0.0, 1.0, seq_len, device=x.device, dtype=x.dtype)
        q_feats = torch.stack([q_pos, 2.0 * q_pos - 1.0], dim=-1).unsqueeze(0).expand(bsz, -1, -1)
        queries = self.query_mlp(q_feats)
        attn_out, _ = self.cross_attn(queries, enc, enc, need_weights=False)
        h = self.norm_q(queries + attn_out)
        return self.out_head(h).squeeze(-1)


class ScaleNormalizedPolySortWrapper(nn.Module):
    """Explicit NON-POLYNOMIAL ablation (A6): wraps a pure PolySortNet with max-abs scaling.

    Divides input by ``s = max_i |x_i| + 1e-8`` before sorting and multiplies output by ``s``,
    eliminating the polynomial scale blow-up cliff at the cost of breaking pure polynomiality.
    """

    def __init__(self, inner: PolySortNet) -> None:
        super().__init__()
        self.inner = inner
        self.n = inner.n
        self.is_pure_polynomial = False
        self.supports_variable_length = inner.supports_variable_length
        self.effective_degree = None
        self.seq_mult_depth = inner.seq_mult_depth

    def for_length(self, new_n: int) -> "ScaleNormalizedPolySortWrapper":
        return ScaleNormalizedPolySortWrapper(self.inner.for_length(new_n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        return self.inner(x / scale) * scale


# ---------------------------------------------------------------------------
# Unified Factory Registry
# ---------------------------------------------------------------------------


def build_model(model_name: str, n: int, **kwargs: Any) -> nn.Module:
    """Instantiate any preregistered model, baseline, control, or ablation by canonical name."""
    if model_name == "polysort_shared":
        return PolySortNet(n=n, shared=True, num_layers=3, num_terms=1, **kwargs)
    if model_name == "polysort_shared_m2":
        return PolySortNet(n=n, shared=True, num_layers=3, num_terms=2, **kwargs)
    if model_name == "polysort_adaptive":
        return PolySortNet(n=n, shared=False, num_layers=3, num_terms=1, **kwargs)
    if model_name == "polysort_adaptive_m2":
        return PolySortNet(n=n, shared=False, num_layers=3, num_terms=2, **kwargs)
    if model_name == "polysort_custom":
        return PolySortNet(n=n, **kwargs)
    if model_name == "polysort_learned_wire":
        return PolySortNetLearnedWire(n=n, **kwargs)
    if model_name == "poly_pairwise_rank":
        return PolyPairwiseRank(n=n, **kwargs)
    if model_name == "deepsets_poly":
        return DeepSetsPoly(n=n, **kwargs)
    if model_name == "pinet_ncp_skip":
        return PiNetSorter(n=n, **kwargs)
    if model_name == "poly_mlp":
        return PolyMLP(n=n, **kwargs)
    if model_name == "analytical_cheon_f1":
        return build_analytical_cheon_sort(n=n, num_layers=kwargs.get("num_layers", 3), num_terms=1)
    if model_name == "analytical_cheon_f2":
        return build_analytical_cheon_sort(n=n, num_layers=kwargs.get("num_layers", 2), num_terms=2)
    if model_name == "analytical_cheon_fg":
        return build_analytical_cheon_fg_sort(n=n, wiring_type=kwargs.get("wiring_type", "bitonic"))
    if model_name == "analytical_remez":
        return build_analytical_remez_sort(
            n=n,
            num_layers=kwargs.get("num_layers", 3),
            num_terms=kwargs.get("num_terms", 1),
            delta=kwargs.get("delta", 0.10),
        )
    if model_name == "identity_control":
        return IdentityControl(n=n)
    if model_name == "linear_control":
        return LinearControl(n=n)
    if model_name == "diffsort_logistic":
        return DiffSortBaseline(n=n, distribution="logistic", **kwargs)
    if model_name == "diffsort_cauchy":
        return DiffSortBaseline(n=n, distribution="cauchy", **kwargs)
    if model_name == "diffsort_reciprocal":
        return DiffSortBaseline(n=n, distribution="reciprocal", **kwargs)
    if model_name == "relu_mlp":
        return ReLUMLPBaseline(n=n, **kwargs)
    if model_name == "set_transformer":
        return SetTransformerBaseline(n=n, **kwargs)
    if model_name == "polysort_scale_norm":
        inner = PolySortNet(n=n, shared=kwargs.pop("shared", False), **kwargs)
        return ScaleNormalizedPolySortWrapper(inner)
    raise ValueError(f"Unknown model_name: {model_name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
