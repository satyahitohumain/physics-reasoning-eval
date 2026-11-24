import json
from pathlib import Path

import pytest
import sympy as sp

from physeval.cli import main
from physeval.grade import build_prompt, grade, judge_prompt, parse_judge_reply
from physeval.problems import load_all

P = load_all()
ROOT = Path(__file__).resolve().parent.parent


def test_every_problem_loads_with_ten_points():
    assert len(P) >= 8
    for p in P.values():
        assert p.max_points == 10, p.id
        assert "FINAL ANSWER" in build_prompt(p)


@pytest.mark.parametrize("pid", sorted(P))
def test_reference_passes_its_own_grader(pid):
    a = P[pid].answer
    final = repr(a.value) if a.type == "numeric" else sp.sstr(a.expr)
    r = grade(P[pid], f"FINAL ANSWER: {final}")
    assert r.final_correct, r.verdict
    assert r.dimension_ok is not False


@pytest.mark.parametrize(
    "pid, wrong",
    [
        ("PHY-001", "sqrt(g/R - w^2)"),
        ("PHY-001", "w"),
        ("PHY-003", "0.4168"),
        ("PHY-004", "q^2 a d/(4 pi epsilon_0 (d^2-a^2)^2)"),  # grounded sphere
        ("PHY-005", "dv*sqrt(r0^3/(G*M))"),  # amplitude, not peak to peak
        ("PHY-006", "2*pi*sqrt(3*(R-r)/(2*g))"),  # fixed drum
        ("PHY-007", "E0*(1+beta)/(1-beta)"),  # Thomson limit only
        ("PHY-008", "m*a^2*V0^2/(8*pi^2*hbar^2)"),  # sign
    ],
)
def test_common_errors_are_rejected(pid, wrong):
    assert not grade(P[pid], f"FINAL ANSWER: {wrong}").final_correct


def test_unknown_symbols_rejected():
    r = grade(P["PHY-001"], "FINAL ANSWER: sqrt(w^2 - g^2/(R^2 w^2)) + 0*z")
    # 0*z simplifies away, so this is still correct
    assert r.final_correct
    r = grade(P["PHY-001"], "FINAL ANSWER: sqrt(w^2 - k)")
    assert not r.final_correct and "symbols" in r.verdict


def test_dimension_flag_on_wrong_answer():
    pytest.importorskip("dimcheck")
    r = grade(P["PHY-001"], "FINAL ANSWER: sqrt(w^2 - g/R^2)")
    assert r.dimension_ok is False


def test_judge_integration():
    p = P["PHY-006"]
    calls = []

    def fake_judge(prompt):
        calls.append(prompt)
        return 'Sure. {"constraint": 2, "conservation": 1.5, "effective_inertia": 99}'

    r = grade(p, "FINAL ANSWER: 2*pi*sqrt((R-r)*(3*M+m)/(g*(2*M+m)))", judge=fake_judge)
    assert r.fully_scored
    assert r.item_scores == {"constraint": 2, "conservation": 1.5, "effective_inertia": 2, "final": 3}
    assert r.auto_points == 8.5
    assert "do not\naward or deduct points for it" in calls[0]
    assert '"final"' not in calls[0].split("RUBRIC")[1].split("RESPONSE")[0]


def test_judge_reply_missing_item():
    with pytest.raises(ValueError):
        parse_judge_reply(P["PHY-006"], '{"constraint": 2}')


def test_judge_prompt_mentions_common_errors():
    assert "fixed-surface" in judge_prompt(P["PHY-006"], "x") or "drum as fixed" in judge_prompt(P["PHY-006"], "x")


def test_cli_grade_demo(tmp_path, capsys):
    out = tmp_path / "results.jsonl"
    assert main(["grade", str(ROOT / "examples" / "demo_responses.jsonl"), "-o", str(out)]) == 0
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    careful = [r for r in rows if r["model"] == "demo-careful"]
    sloppy = [r for r in rows if r["model"] == "demo-sloppy"]
    assert all(r["final_correct"] for r in careful)
    assert not any(r["final_correct"] for r in sloppy)
    assert "| demo-careful |" in capsys.readouterr().out


def test_cli_validate():
    assert main(["validate"]) == 0
