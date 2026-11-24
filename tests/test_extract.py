import pytest
import sympy as sp

from physeval.extract import ExtractionError, final_answer_text, latex_to_plain, to_number, to_sympy
from physeval.problems import load_all

P = load_all()


def parse(pid, text):
    a = P[pid].answer
    return to_sympy(text, a.symbols, a.aliases, a.definitions)


def same(pid, text):
    return sp.simplify(parse(pid, text) - P[pid].answer.expr) == 0


def test_final_answer_line_wins_over_boxed():
    r = "blah \\boxed{x} more\nFINAL ANSWER: y"
    assert final_answer_text(r) == "y"


def test_last_boxed_with_nested_braces():
    r = r"first \boxed{a} then \boxed{\frac{1}{2}}"
    assert final_answer_text(r) == r"\frac{1}{2}"


def test_no_answer():
    assert final_answer_text("I am not sure.") is None


@pytest.mark.parametrize(
    "pid, text",
    [
        ("PHY-001", r"\sqrt{\omega^2 - \frac{g^2}{R^2\omega^2}}"),
        ("PHY-001", r"\Omega = \omega\sqrt{1-\left(\frac{g}{R\omega^2}\right)^2}"),
        ("PHY-001", "sqrt(w^2 - g^2/(R^2 w^2))"),
        ("PHY-002", r"E = \frac{3}{2}\sqrt[3]{\frac{F^2\hbar^2}{2\pi m}}"),
        ("PHY-004", r"\frac{q^2 a}{4\pi\varepsilon_0}\left[\frac{d}{(d^2-a^2)^2}-\frac{1}{d^3}\right]"),
        ("PHY-005", r"2\,\delta v\sqrt{\frac{r_0^3}{GM}}"),
        ("PHY-005", "2 dv sqrt(r0^3/(G M))"),
        ("PHY-006", r"2\pi\sqrt{\frac{(R-r)(3M+m)}{g(2M+m)}}"),
        ("PHY-007", r"\frac{\gamma m c^2 E_0 (1+\beta)}{\gamma m c^2 (1-\beta) + 2E_0}"),
        ("PHY-008", r"-\frac{m a^2 V_0^2}{8\pi^2\hbar^2}"),
        ("PHY-008", r"-\dfrac{m a^{2} V_{0}^{2}}{8 \pi^{2} \hbar^{2}}."),
    ],
)
def test_latex_and_plain_forms(pid, text):
    assert same(pid, text)


def test_multi_letter_split():
    # "GM" is not a declared symbol, so it is read as G*M
    e = parse("PHY-005", "2*dv*sqrt(r0**3/(GM))")
    assert {str(s) for s in e.free_symbols} == {"dv", "r0", "G", "M"}


@pytest.mark.parametrize(
    "text, value",
    [("0.3515", 0.3515), (r"k_B T/\Delta \approx 0.3515", 0.3515), (r"3.515\times 10^{-1}", 0.3515), ("≈ 0.35", 0.35)],
)
def test_numbers(text, value):
    assert to_number(text) == pytest.approx(value)


def test_unsupported_command():
    with pytest.raises(ExtractionError):
        latex_to_plain(r"\weird{x}")


def test_garbage_raises():
    with pytest.raises(ExtractionError):
        parse("PHY-001", "sqrt((")
