"""Pull the final answer out of a free-text response and turn it into a SymPy expression.

Responses are asked to end with ``FINAL ANSWER: <expression>``; as a fallback the last
``\\boxed{...}`` is used. The expression may be plain text (``sqrt(w**2 - g/R)``) or LaTeX
(``\\sqrt{\\omega^2 - \\frac{g}{R}}``). LaTeX is translated with a small, dependency-free
converter that covers the notation physics answers actually use.
"""

from __future__ import annotations

import re
from typing import Iterable

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    split_symbols_custom,
    standard_transformations,
)

_FINAL = re.compile(r"final\s+answer\s*[:=]\s*(.+)", re.IGNORECASE)

GREEK = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
    "lambda", "mu", "nu", "xi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Sigma", "Upsilon", "Phi", "Psi", "Omega",
]
COMMAND_MAP = {
    "hbar": "hbar", "pi": "pi", "varepsilon": "epsilon", "epsilon": "epsilon", "varphi": "phi",
    "vartheta": "theta", "ell": "l", "infty": "oo",
    "sin": "sin", "cos": "cos", "tan": "tan", "exp": "exp", "ln": "log", "log": "log",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "arcsin": "asin", "arccos": "acos", "arctan": "atan",
    "cdot": "*", "times": "*",
}
FUNCTIONS = {"sin", "cos", "tan", "exp", "log", "sqrt", "sinh", "cosh", "tanh", "asin", "acos", "atan", "Abs"}
_RESERVED = {"lambda": "lamda"}  # 'lambda' is a Python keyword


class ExtractionError(ValueError):
    pass


def final_answer_text(response: str) -> str | None:
    """Return the raw text of the final answer, or ``None`` if none is found."""
    lines = [m.group(1).strip() for m in _FINAL.finditer(response)]
    if lines:
        return lines[-1].strip().strip("$").strip()
    boxed = _last_boxed(response)
    return boxed.strip() if boxed is not None else None


def _last_boxed(text: str) -> str | None:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    content, end = _braced(text, i - 1)
    return content


def _braced(text: str, open_idx: int) -> tuple[str, int]:
    """Given the index of '{', return (content, index after matching '}')."""
    if open_idx >= len(text) or text[open_idx] != "{":
        raise ExtractionError(f"expected '{{' at position {open_idx}")
    depth = 0
    for j in range(open_idx, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : j], j + 1
    raise ExtractionError("unbalanced braces")


def _arg(text: str, i: int) -> tuple[str, int]:
    """Read one LaTeX argument at position i: a braced group or a single character."""
    while i < len(text) and text[i] == " ":
        i += 1
    if i < len(text) and text[i] == "{":
        return _braced(text, i)
    if i < len(text) and text[i] == "\\":
        m = re.match(r"\\[A-Za-z]+", text[i:])
        if m:
            return text[i : i + m.end()], i + m.end()
    return text[i : i + 1], i + 1


def latex_to_plain(tex: str) -> str:
    """Translate common LaTeX math into SymPy-parsable text.

    >>> latex_to_plain(r"\\frac{g}{R_{0}}")
    '((g)/(R_0))'
    """
    s = tex
    for junk in (r"\displaystyle", r"\left", r"\right", r"\,", r"\;", r"\!", r"\:", r"\quad", r"\qquad", "$"):
        s = s.replace(junk, " " if junk.startswith("\\") and len(junk) > 2 else "")
    s = re.sub(r"\\(?:mathrm|text|operatorname|mathit)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[dD]elta\s*([A-Za-z])(?![A-Za-z])", r"delta_\1", s)  # \delta v -> delta_v
    s = re.sub(r"\\[,;!]", " ", s)
    return _convert(s).strip()


def _convert(s: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+)", s[i:])
            if not m:
                i += 2
                continue
            name = m.group(1)
            i += m.end()
            if name in ("frac", "dfrac", "tfrac"):
                num, i = _arg(s, i)
                den, i = _arg(s, i)
                out.append(f"(({_convert(num)})/({_convert(den)}))")
            elif name == "sqrt":
                if i < len(s) and s[i] == "[":
                    close = s.index("]", i)
                    index = _convert(s[i + 1 : close])
                    i = close + 1
                    body, i = _arg(s, i)
                    out.append(f"(({_convert(body)})**(1/({index})))")
                else:
                    body, i = _arg(s, i)
                    out.append(f" sqrt({_convert(body)})")
            elif name in COMMAND_MAP:
                out.append(f" {COMMAND_MAP[name]} ")
            elif name in GREEK:
                out.append(f" {_RESERVED.get(name, name)} ")
            else:
                raise ExtractionError(f"unsupported LaTeX command \\{name}")
        elif c == "^":
            arg, i = _arg(s, i + 1)
            out.append(f"**({_convert(arg)})")
        elif c == "_":
            arg, i = _arg(s, i + 1)
            sub = re.sub(r"[^A-Za-z0-9]", "", _convert(arg))
            # Attach subscript to the preceding identifier: r_{0} -> r_0
            while out and out[-1] == " ":
                out.pop()
            if out:
                out[-1] = out[-1].rstrip() + "_" + sub
            else:
                out.append("_" + sub)
        elif c in "{}":
            out.append("(" if c == "{" else ")")
            i += 1
        elif c in "[]":
            out.append("(" if c == "[" else ")")
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _looks_like_latex(text: str) -> bool:
    return "\\" in text or re.search(r"\^\{|_\{", text) is not None


def _clean_rhs(text: str) -> str:
    """Drop a leading ``Omega =`` and anything after an approximation sign."""
    text = text.strip().rstrip(".").strip()
    parts = [p for p in re.split(r"\\approx|≈|\\simeq", text) if p.strip()]
    text = parts[0] if parts else ""
    if "=" in text:
        text = text.split("=")[-1]
    return text.strip()


def to_sympy(
    text: str,
    known: Iterable[str],
    aliases: dict[str, str] | None = None,
    definitions: dict[str, sp.Expr] | None = None,
) -> sp.Expr:
    """Parse an answer string into SymPy using ``known`` symbol names (all positive reals).

    Unknown multi-letter names are split into single letters (``GM`` -> ``G*M``) unless they
    are known symbols, aliases or functions.
    """
    aliases = aliases or {}
    known = set(known)
    definitions = definitions or {}
    names = known | set(aliases) | set(definitions)
    text = _clean_rhs(text)
    plain = latex_to_plain(text) if _looks_like_latex(text) else text
    plain = plain.replace("×", "*").replace("·", "*").replace("−", "-").replace("π", "pi")
    plain = re.sub(r"(\d)(?![eE][-+]?\d)([A-Za-z_])", r"\1 \2", plain)  # 2E_0 -> 2 E_0
    plain = re.sub(r"(\d)\s*\*\s*10\s*\*\*\s*\(?\s*([-+]?\d+)\s*\)?", r"\1e\2", plain)

    def can_split(token: str) -> bool:
        return token not in names and token not in FUNCTIONS and token not in GREEK and token not in ("pi", "hbar", "oo", "lamda")

    local = {n: sp.Symbol(n, positive=True) for n in names}
    for alias, target in aliases.items():
        local[alias] = sp.Symbol(target, positive=True)
    local.update({"pi": sp.pi, "e": sp.E, "E": sp.Symbol("E", positive=True) if "E" in names else sp.E})
    transformations = standard_transformations + (
        convert_xor,
        split_symbols_custom(can_split),
        implicit_multiplication_application,
    )
    try:
        expr = parse_expr(plain, local_dict=local, transformations=transformations)
    except Exception as exc:  # tokenizer/sympify errors
        raise ExtractionError(f"could not parse answer {text!r} (as {plain!r}): {exc}") from exc
    if not isinstance(expr, sp.Expr):
        raise ExtractionError(f"answer {text!r} is not an expression")
    if definitions:
        expr = expr.subs({sp.Symbol(k, positive=True): v for k, v in definitions.items()})
    return expr


_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def to_number(text: str) -> float:
    """Parse a numeric final answer such as ``0.3515``, ``3.515\\times10^{-1}`` or ``≈ 0.35``."""
    parts = [p for p in re.split(r"\\approx|≈|\\simeq", text) if p.strip()]
    text = parts[-1] if parts else text
    text = _clean_rhs(text)
    plain = latex_to_plain(text) if _looks_like_latex(text) else text
    plain = plain.replace("×", "*").replace(" ", "")
    m = re.fullmatch(r"([-+]?\d*\.?\d+)\*10\*\*\(?([-+]?\d+)\)?", plain)
    if m:
        return float(m.group(1)) * 10 ** int(m.group(2))
    try:
        return float(sp.N(parse_expr(plain)))
    except Exception:
        found = _NUMBER.findall(plain)
        if not found:
            raise ExtractionError(f"no number in answer {text!r}") from None
        return float(found[0])
