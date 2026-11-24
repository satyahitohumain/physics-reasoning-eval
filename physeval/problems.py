"""Loading and validating problem files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sympy as sp
import yaml

PROBLEM_DIR = Path(__file__).resolve().parent.parent / "problems"
REQUIRED = ("id", "title", "domain", "difficulty", "statement", "answer", "rubric")


class ProblemError(ValueError):
    pass


@dataclass(frozen=True)
class RubricItem:
    id: str
    points: float
    criterion: str
    auto: str | None = None  # "final_answer" for items graded automatically


@dataclass
class Answer:
    type: str  # "symbolic" or "numeric"
    expr: sp.Expr | None = None
    value: float | None = None
    rel_tol: float = 1e-6
    symbols: dict[str, sp.Symbol] = field(default_factory=dict)
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    constraints: list[sp.Expr] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)
    definitions: dict[str, sp.Expr] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    answer_unit: str | None = None


@dataclass
class Problem:
    id: str
    title: str
    domain: str
    difficulty: int
    statement: str
    answer: Answer
    rubric: list[RubricItem]
    solution_sketch: str = ""
    common_errors: list[str] = field(default_factory=list)

    @property
    def max_points(self) -> float:
        return sum(item.points for item in self.rubric)


def _parse_answer(raw: dict[str, Any], pid: str) -> Answer:
    kind = raw.get("type")
    if kind == "numeric":
        return Answer(type="numeric", value=float(raw["value"]), rel_tol=float(raw.get("rel_tol", 1e-3)))
    if kind != "symbolic":
        raise ProblemError(f"{pid}: answer.type must be 'symbolic' or 'numeric', got {kind!r}")
    spec = raw.get("symbols") or {}
    symbols = {name: sp.Symbol(name, positive=True) for name in spec}
    ranges = {}
    for name, opts in spec.items():
        lo, hi = (opts or {}).get("range", [0.5, 2.0])
        ranges[name] = (float(lo), float(hi))
    local = dict(symbols)
    try:
        expr = sp.sympify(raw["expr"], locals=local)
        constraints = [sp.sympify(c, locals=local) for c in raw.get("constraints", [])]
        definitions = {str(k): sp.sympify(v, locals=local) for k, v in (raw.get("definitions") or {}).items()}
    except (sp.SympifyError, KeyError) as exc:
        raise ProblemError(f"{pid}: cannot parse answer: {exc}") from exc
    extra = {str(s) for s in expr.free_symbols} - set(symbols)
    if extra:
        raise ProblemError(f"{pid}: answer uses undeclared symbols {sorted(extra)}")
    return Answer(
        type="symbolic",
        expr=expr,
        symbols=symbols,
        ranges=ranges,
        constraints=constraints,
        aliases={str(k): str(v) for k, v in (raw.get("aliases") or {}).items()},
        definitions=definitions,
        units={str(k): str(v) for k, v in (raw.get("units") or {}).items()},
        answer_unit=raw.get("answer_unit"),
    )


def load_problem(path: str | Path) -> Problem:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        raise ProblemError(f"{path.name}: missing fields {missing}")
    rubric = [
        RubricItem(id=str(r["id"]), points=float(r["points"]), criterion=str(r["criterion"]), auto=r.get("auto"))
        for r in data["rubric"]
    ]
    ids = [r.id for r in rubric]
    if len(ids) != len(set(ids)):
        raise ProblemError(f"{data['id']}: duplicate rubric ids")
    if not any(r.auto == "final_answer" for r in rubric):
        raise ProblemError(f"{data['id']}: rubric needs one item with auto: final_answer")
    return Problem(
        id=str(data["id"]),
        title=str(data["title"]),
        domain=str(data["domain"]),
        difficulty=int(data["difficulty"]),
        statement=str(data["statement"]).strip(),
        answer=_parse_answer(data["answer"], data["id"]),
        rubric=rubric,
        solution_sketch=str(data.get("solution_sketch", "")).strip(),
        common_errors=list(data.get("common_errors", [])),
    )


def load_all(directory: str | Path = PROBLEM_DIR) -> dict[str, Problem]:
    problems = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        p = load_problem(path)
        if p.id in problems:
            raise ProblemError(f"duplicate problem id {p.id}")
        problems[p.id] = p
    return problems
