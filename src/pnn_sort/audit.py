"""Automated PyTorch FX pure-polynomial graph auditor and algebraic invariant checks.

Mechanically guarantees that every model claiming ``is_pure_polynomial = True``
executes EXCLUSIVELY polynomial operations (additions, subtractions, linear maps,
multiplications, non-negative integer powers, and constant scalar scaling) with
zero non-polynomial activations, divisions by tensors, or sorting primitives.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import List, Set

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

ALLOWED_FUNCTIONS: Set[object] = {
    operator.add,
    operator.sub,
    operator.mul,
    operator.neg,
    operator.pos,
    operator.matmul,
    operator.getitem,
    torch.add,
    torch.sub,
    torch.mul,
    torch.matmul,
    torch.mm,
    torch.bmm,
    torch.einsum,
    torch.sum,
    torch.mean,
    torch.cat,
    torch.stack,
    torch.ones_like,
    torch.zeros_like,
    torch.unsqueeze,
    torch.squeeze,
    torch.transpose,
    torch.permute,
    torch.reshape,
    torch.flatten,
    F.linear,
}

ALLOWED_METHODS: Set[str] = {
    "add",
    "sub",
    "mul",
    "matmul",
    "sum",
    "mean",
    "transpose",
    "permute",
    "reshape",
    "view",
    "unsqueeze",
    "squeeze",
    "expand",
    "expand_as",
    "contiguous",
    "clone",
    "to",
    "type",
    "float",
    "double",
    "size",
    "shape",
}

ALLOWED_MODULE_TYPES = (nn.Linear,)


@dataclass
class PurityAuditResult:
    passed: bool
    violations: List[str]
    node_count: int
    op_counts: dict[str, int]


def _is_const_nonzero_scalar(arg: object) -> bool:
    return isinstance(arg, (int, float)) and arg != 0


def audit_pure_polynomial(model: nn.Module) -> PurityAuditResult:
    """Symbolically trace ``model`` with PyTorch FX and verify strict polynomial purity."""
    model.eval()
    traced = fx.symbolic_trace(model)
    violations: List[str] = []
    op_counts: dict[str, int] = {}

    for node in traced.graph.nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        if node.op == "call_function":
            fn = node.target
            fn_name = getattr(fn, "__name__", str(fn))
            op_counts[fn_name] = op_counts.get(fn_name, 0) + 1
            if fn in ALLOWED_FUNCTIONS:
                continue
            if fn in (operator.truediv, torch.div):
                # Allow division ONLY by a compile-time constant scalar (equivalent to scalar mul).
                if len(node.args) == 2 and _is_const_nonzero_scalar(node.args[1]):
                    continue
                violations.append(f"Non-constant division node: {node.name} args={node.args}")
            elif fn in (operator.pow, torch.pow):
                if len(node.args) == 2 and isinstance(node.args[1], int) and node.args[1] >= 0:
                    continue
                violations.append(f"Non-polynomial power node: {node.name} args={node.args}")
            else:
                violations.append(f"Forbidden function op: {fn_name} ({node.name})")
        elif node.op == "call_method":
            meth = str(node.target)
            op_counts[f".{meth}"] = op_counts.get(f".{meth}", 0) + 1
            if meth in ALLOWED_METHODS:
                continue
            if meth == "pow" and len(node.args) == 2 and isinstance(node.args[1], int) and node.args[1] >= 0:
                continue
            if meth == "div" and len(node.args) == 2 and _is_const_nonzero_scalar(node.args[1]):
                continue
            violations.append(f"Forbidden method op: .{meth} ({node.name})")
        elif node.op == "call_module":
            submod = traced.get_submodule(str(node.target))
            cls_name = submod.__class__.__name__
            op_counts[cls_name] = op_counts.get(cls_name, 0) + 1
            if not isinstance(submod, ALLOWED_MODULE_TYPES):
                violations.append(f"Forbidden module op: {cls_name} ({node.target})")

    return PurityAuditResult(
        passed=len(violations) == 0,
        violations=violations,
        node_count=len(traced.graph.nodes),
        op_counts=op_counts,
    )
