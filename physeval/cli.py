"""Command line: ``physeval list | prompts | grade | validate``."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import sympy as sp

from .grade import GradeResult, build_prompt, command_judge, grade
from .problems import PROBLEM_DIR, load_all


def _report(results: list[GradeResult]) -> str:
    by_model: dict[str, list[GradeResult]] = defaultdict(list)
    for r in results:
        by_model[r.model].append(r)
    lines = [
        "| model | problems | final answer correct | rubric points | dimension errors |",
        "|---|---|---|---|---|",
    ]
    for model, rs in sorted(by_model.items()):
        n = len(rs)
        correct = sum(r.final_correct for r in rs)
        if all(r.fully_scored for r in rs):
            pts = f"{sum(r.auto_points for r in rs):g} / {sum(r.max_points for r in rs):g}"
        else:
            pts = "needs judge"
        dim_bad = sum(r.dimension_ok is False for r in rs)
        lines.append(f"| {model} | {n} | {correct}/{n} ({100 * correct / n:.0f}%) | {pts} | {dim_bad} |")
    lines += ["", "| problem | model | correct | verdict |", "|---|---|---|---|"]
    for r in sorted(results, key=lambda r: (r.problem_id, r.model)):
        verdict = r.verdict.replace("|", "\\|")
        lines.append(f"| {r.problem_id} | {r.model} | {'yes' if r.final_correct else 'no'} | {verdict} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="physeval", description="Hard physics problems with verifiable answers and rubrics.")
    ap.add_argument("--problems", default=str(PROBLEM_DIR), help="directory of problem YAML files")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list problems")
    p = sub.add_parser("prompts", help="write one prompt per problem as JSONL")
    p.add_argument("-o", "--out", default="-")
    p = sub.add_parser("grade", help="grade a JSONL file of responses")
    p.add_argument("responses", help='JSONL with fields "problem_id", "model", "response"')
    p.add_argument("-o", "--out", help="write per-response results as JSONL")
    p.add_argument("--judge-cmd", help="shell command that reads a grading prompt on stdin and prints JSON scores")
    sub.add_parser("validate", help="check that every reference answer passes its own grader")
    args = ap.parse_args(argv)

    problems = load_all(args.problems)

    if args.cmd == "list":
        for p in problems.values():
            print(f"{p.id}  [{p.domain}, difficulty {p.difficulty}]  {p.title}")
        return 0

    if args.cmd == "prompts":
        out = sys.stdout if args.out == "-" else open(args.out, "w", encoding="utf-8")
        for p in problems.values():
            out.write(json.dumps({"problem_id": p.id, "prompt": build_prompt(p)}) + "\n")
        if out is not sys.stdout:
            out.close()
        return 0

    if args.cmd == "validate":
        failed = 0
        for p in problems.values():
            a = p.answer
            final = repr(a.value) if a.type == "numeric" else sp.sstr(a.expr)
            r = grade(p, f"FINAL ANSWER: {final}", model="reference")
            ok = r.final_correct and r.dimension_ok is not False
            failed += not ok
            dim = {True: "dims ok", False: "DIMENSION MISMATCH", None: "dims n/a"}[r.dimension_ok]
            print(f"{'ok  ' if ok else 'FAIL'} {p.id}  {dim}  {r.verdict}")
        return 1 if failed else 0

    if args.cmd == "grade":
        judge = command_judge(args.judge_cmd) if args.judge_cmd else None
        results = []
        for lineno, line in enumerate(Path(args.responses).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            pid = row["problem_id"]
            if pid not in problems:
                print(f"line {lineno}: unknown problem {pid!r}", file=sys.stderr)
                continue
            results.append(grade(problems[pid], row["response"], row.get("model", "unknown"), judge))
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r.to_dict()) + "\n")
        print(_report(results))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
