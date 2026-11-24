"""Grade free-text responses against a problem's reference answer and rubric."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Callable, Protocol

import sympy as sp

from .equivalence import Verdict, numeric_equal, symbolic_equal
from .extract import ExtractionError, final_answer_text, to_number, to_sympy
from .problems import Problem

ANSWER_INSTRUCTIONS = (
    "Solve the problem. Show your reasoning. On the last line write exactly\n"
    "FINAL ANSWER: <expression>\n"
    "using only the symbols given in the problem (plain text like sqrt(g/R) or LaTeX), "
    "with no units."
)


def build_prompt(problem: Problem) -> str:
    return f"{problem.statement}\n\n{ANSWER_INSTRUCTIONS}"


class Judge(Protocol):
    def __call__(self, prompt: str) -> str: ...


@dataclass
class GradeResult:
    problem_id: str
    model: str
    extracted: str | None
    parsed: str | None
    final_correct: bool
    verdict: str
    dimension_ok: bool | None
    item_scores: dict[str, float | None] = field(default_factory=dict)
    max_points: float = 0.0

    @property
    def auto_points(self) -> float:
        return sum(v for v in self.item_scores.values() if v is not None)

    @property
    def fully_scored(self) -> bool:
        return all(v is not None for v in self.item_scores.values())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["points"] = self.auto_points if self.fully_scored else None
        d["auto_points"] = self.auto_points
        return d


def _dimension_ok(problem: Problem, expr: sp.Expr) -> bool | None:
    """Return whether ``expr`` has the answer's dimension, or None if it can't be checked."""
    a = problem.answer
    if not a.units or not a.answer_unit:
        return None
    try:
        from dimcheck import dimension_of, parse_dimension  # optional dependency
    except ImportError:
        return None
    try:
        return dimension_of(expr, a.units) == parse_dimension(a.answer_unit)
    except Exception:
        return False


def judge_prompt(problem: Problem, response: str) -> str:
    items = [r for r in problem.rubric if r.auto is None]
    rubric = "\n".join(f'- "{r.id}" (0 to {r.points:g} points): {r.criterion}' for r in items)
    errors = "\n".join(f"- {e}" for e in problem.common_errors) or "- (none listed)"
    keys = ", ".join(f'"{r.id}": <points>' for r in items)
    return f"""You are grading a physics solution against a rubric. Grade only the reasoning steps
listed below. The final answer is checked separately by a computer algebra system, so do not
award or deduct points for it.

PROBLEM
{problem.statement}

REFERENCE SOLUTION (for the grader only)
{problem.solution_sketch}

COMMON ERRORS TO WATCH FOR
{errors}

RUBRIC
{rubric}

Award an item only if the response clearly does what it describes; equivalent methods count.
Award partial credit only where the item has more than one point and the work is partly right.

RESPONSE TO GRADE
<<<
{response}
>>>

Reply with a single JSON object and nothing else: {{{keys}}}"""


def parse_judge_reply(problem: Problem, reply: str) -> dict[str, float]:
    match = re.search(r"\{.*\}", reply, re.S)
    if not match:
        raise ValueError(f"judge reply contains no JSON object: {reply[:200]!r}")
    data = json.loads(match.group(0))
    scores = {}
    for item in problem.rubric:
        if item.auto is not None:
            continue
        raw = data.get(item.id)
        if raw is None:
            raise ValueError(f"judge reply is missing rubric item {item.id!r}")
        scores[item.id] = max(0.0, min(float(raw), item.points))
    return scores


def command_judge(command: str, timeout: float = 300) -> Judge:
    """A judge that pipes the prompt to a shell command and reads its stdout.

    Any model can be used this way, e.g. ``--judge-cmd "llm -m some-model"``.
    """

    def run(prompt: str) -> str:
        proc = subprocess.run(shlex.split(command), input=prompt, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"judge command failed ({proc.returncode}): {proc.stderr.strip()[:300]}")
        return proc.stdout

    return run


def grade(problem: Problem, response: str, model: str = "unknown", judge: Callable[[str], str] | None = None) -> GradeResult:
    a = problem.answer
    text = final_answer_text(response)
    parsed: str | None = None
    dim_ok: bool | None = None
    if text is None:
        verdict = Verdict(False, "extraction", "no 'FINAL ANSWER:' line or \\boxed{} found")
    else:
        try:
            if a.type == "numeric":
                value = to_number(text)
                parsed = repr(value)
                verdict = numeric_equal(value, a)
            else:
                expr = to_sympy(text, a.symbols, a.aliases, a.definitions)
                parsed = str(expr)
                verdict = symbolic_equal(expr, a)
                dim_ok = _dimension_ok(problem, expr)
        except ExtractionError as exc:
            verdict = Verdict(False, "parse", str(exc))

    scores: dict[str, float | None] = {}
    for item in problem.rubric:
        if item.auto == "final_answer":
            scores[item.id] = item.points if verdict.correct else 0.0
        else:
            scores[item.id] = None
    if judge is not None:
        scores.update(parse_judge_reply(problem, judge(judge_prompt(problem, response))))

    return GradeResult(
        problem_id=problem.id,
        model=model,
        extracted=text,
        parsed=parsed,
        final_correct=verdict.correct,
        verdict=f"{verdict.method}: {verdict.detail}".rstrip(": "),
        dimension_ok=dim_ok,
        item_scores=scores,
        max_points=problem.max_points,
    )
