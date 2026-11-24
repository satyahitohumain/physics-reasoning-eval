"""Decide whether a candidate expression equals the reference answer."""

from __future__ import annotations

import random
from dataclasses import dataclass

import sympy as sp

from .problems import Answer


@dataclass
class Verdict:
    correct: bool
    method: str
    detail: str = ""


def _sample_point(answer: Answer, rng: random.Random, tries: int = 200) -> dict[sp.Symbol, float]:
    for _ in range(tries):
        point = {sym: rng.uniform(*answer.ranges[name]) for name, sym in answer.symbols.items()}
        if all(float(c.subs(point)) > 0 for c in answer.constraints):
            return point
    raise RuntimeError("could not sample a point satisfying the problem's constraints")


def _fmt(z: complex) -> str:
    return f"{z.real:.6g}" if abs(z.imag) <= 1e-12 * max(abs(z.real), 1e-300) else f"{z:.6g}"


def symbolic_equal(candidate: sp.Expr, answer: Answer, samples: int = 24, rel_tol: float = 1e-7, seed: int = 0) -> Verdict:
    """Compare ``candidate`` with the reference.

    The fast path is ``simplify(candidate - reference) == 0``. Because ``simplify`` can miss
    true identities (and is slow on large expressions), the decisive test is numerical:
    both expressions are evaluated at random points inside the problem's physical domain,
    and must agree to ``rel_tol`` at every point.
    """
    ref = answer.expr
    assert ref is not None
    unknown = {str(s) for s in candidate.free_symbols} - set(answer.symbols)
    if unknown:
        return Verdict(False, "symbols", f"answer uses symbols not in the problem: {sorted(unknown)}")
    try:
        if sp.simplify(candidate - ref) == 0:
            return Verdict(True, "symbolic")
    except Exception:  # pragma: no cover - simplify can raise on exotic input
        pass
    rng = random.Random(seed)
    f_ref = sp.lambdify(list(answer.symbols.values()), ref, "mpmath")
    f_can = sp.lambdify(list(answer.symbols.values()), candidate, "mpmath")
    for _ in range(samples):
        point = _sample_point(answer, rng)
        args = list(point.values())
        try:
            a, b = complex(f_ref(*args)), complex(f_can(*args))
        except (ZeroDivisionError, ValueError, TypeError, OverflowError) as exc:
            return Verdict(False, "numeric", f"candidate cannot be evaluated: {exc}")
        scale = max(abs(a), abs(b), 1e-300)
        if abs(a - b) / scale > rel_tol:
            pretty = ", ".join(f"{k}={v:.4g}" for k, v in point.items())
            return Verdict(False, "numeric", f"differs at {pretty}: reference {_fmt(a)}, candidate {_fmt(b)}")
    return Verdict(True, "numeric", f"agrees at {samples} random points")


def numeric_equal(value: float, answer: Answer) -> Verdict:
    assert answer.value is not None
    err = abs(value - answer.value) / abs(answer.value)
    ok = err <= answer.rel_tol
    return Verdict(ok, "numeric", f"relative error {err:.2e} (tolerance {answer.rel_tol:.0e})")
