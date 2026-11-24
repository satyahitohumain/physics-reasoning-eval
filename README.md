# physics-reasoning-eval

[![tests](https://github.com/satyahitohumain/physics-reasoning-eval/actions/workflows/tests.yml/badge.svg)](https://github.com/satyahitohumain/physics-reasoning-eval/actions/workflows/tests.yml)

A small, carefully built set of **original graduate-level physics problems** for evaluating
the reasoning of language models, with a grader that checks final answers by computer algebra
and scores the working against a point rubric.

Every reference answer is **independently re-derived by a numerical method** (direct simulation,
a multipole series, brute-force diagonalisation) and these checks run in CI. If the dataset
says the answer is X, there is a test that recomputes X a different way.

```console
$ physeval grade examples/demo_responses.jsonl
| model | problems | final answer correct | rubric points | dimension errors |
|---|---|---|---|---|
| demo-careful | 6 | 6/6 (100%) | needs judge | 0 |
| demo-sloppy | 6 | 0/6 (0%) | needs judge | 1 |

| problem | model | correct | verdict |
|---|---|---|---|
| PHY-001 | demo-careful | yes | symbolic |
| PHY-001 | demo-sloppy | no | numeric: differs at w=8.6, g=15.4, R=1.378: reference 8.50097, candidate 0+7.92318j |
...
```

The demo responses are hand-written to illustrate typical failure modes, not outputs of any
real model.

## The problems

| id | domain | difficulty | what makes it hard |
|---|---|---|---|
| PHY-001 | classical mechanics | 3 | linearising about a non-trivial equilibrium in a rotating frame |
| PHY-002 | quantum mechanics | 3 | variational method with a non-analytic potential, closed form required |
| PHY-003 | statistical mechanics | 2 | degeneracy shifts the Schottky peak; transcendental condition |
| PHY-004 | electromagnetism | 3 | isolated *neutral* sphere, not the textbook grounded case |
| PHY-005 | classical mechanics | 2 | epicyclic frequency, factor-of-two traps |
| PHY-006 | classical mechanics | 4 | rolling inside a drum that is itself free to turn; cyclic coordinate |
| PHY-007 | special relativity | 3 | exact inverse Compton backscatter, no Thomson approximation |
| PHY-008 | quantum mechanics | 4 | second-order perturbation series summed in closed form ($\pi^2/64$) |

Each problem file in [`problems/`](problems) contains:

- `statement`: the prompt, in Markdown with LaTeX;
- `answer`: a SymPy expression (or a number with tolerance), the physical sampling range of each
  symbol, domain constraints such as `d > a`, accepted aliases (`omega` for `w`), definitions
  (`gamma = 1/sqrt(1 - beta**2)`) and the unit of every symbol;
- `rubric`: 10 points split over the key reasoning steps, with one item graded automatically
  from the final answer;
- `common_errors`: the specific wrong turns seen in practice, which the judge is shown;
- `solution_sketch`: a compact reference derivation, including limiting-case checks.

The problems are written to be **resistant to pattern-matching**: each one is a twist on a
standard setup (a neutral sphere instead of grounded, a free drum instead of a fixed surface, a
degenerate excited level) so that recalling the textbook answer produces a specific, gradable
wrong answer listed in `common_errors`.

## How grading works

1. **Extraction.** The model is asked to end with `FINAL ANSWER: <expr>`; the last `\boxed{}`
   is a fallback. Both plain text (`sqrt(w^2 - g^2/(R^2 w^2))`) and LaTeX
   (`\omega\sqrt{1-\left(\frac{g}{R\omega^2}\right)^2}`) are accepted. LaTeX goes through a small
   built-in converter (fractions, roots of any order, sub/superscripts, Greek letters,
   `\left`/`\right`, `\delta v`), with no ANTLR or other heavy dependency.
2. **Equivalence.** `simplify(candidate - reference) == 0` is tried first. Because `simplify`
   can miss true identities, the decisive test is numerical: both expressions are evaluated at
   24 random points inside the physical domain (respecting constraints like `w^2 R > g`) and must
   agree to 1e-7. Answers that use symbols not in the problem are rejected with a clear reason.
3. **Units.** If [`dimcheck`](https://github.com/satyahitohumain/dimcheck) is installed, the
   parsed answer is also checked for the right physical dimension, and dimension errors are
   counted separately in the report. A dimensionally wrong answer tells you something
   different about a model than a wrong coefficient does.
4. **Rubric.** The final-answer item is always scored by the checker, never by the judge. The
   other items need a judge: pass any command that reads a prompt on stdin and prints JSON,
   e.g. `--judge-cmd "llm -m <model>"`. The judge sees the reference solution, the
   common-error list and only the process items, and its scores are clamped to each item's
   range.

Keeping verifiable checks out of the judge's hands is deliberate: the judge can't
award points for a confident but wrong final expression, or take them away from a correct one.

## Usage

```bash
pip install -e ".[dev]"          # add ".[units]" for dimension checks

physeval list                     # list problems
physeval prompts -o prompts.jsonl # one prompt per problem
# ... run your model, write {"problem_id", "model", "response"} per line ...
physeval grade responses.jsonl -o results.jsonl
physeval grade responses.jsonl --judge-cmd "llm -m <judge-model>"
physeval validate                 # every reference answer passes its own grader

python -m verification.checks     # independent numerical re-derivation of every answer
pytest                            # all of the above, plus extraction and grading tests
```

```text
$ python -m verification.checks
ok   PHY-001  numerical=6.41435823     reference=6.414358256    rel.err=4.0e-09
ok   PHY-002  numerical=0.5960994512   reference=0.5961000152   rel.err=9.5e-07
ok   PHY-003  numerical=0.3514952614   reference=0.3514952      rel.err=1.7e-07
ok   PHY-004  numerical=0.1210896886   reference=0.1210896886   rel.err=2.3e-16
ok   PHY-005  numerical=0.0001999999793 reference=0.0002        rel.err=1.0e-07
ok   PHY-006  numerical=1.959944041    reference=1.95994404     rel.err=6.3e-10
ok   PHY-007  numerical=0.7619047619   reference=0.7619047619   rel.err=0.0e+00
ok   PHY-008  numerical=-0.01266471504 reference=-0.01266514796 rel.err=3.4e-05
```

## Adding a problem

Copy any file in `problems/`, give it a new id, and add a function to
`verification/checks.py` that recomputes the answer without using the closed form. A problem
is accepted when `physeval validate` and `pytest` both pass.

## Related work

This project's problems, grader and checks are original. The design was informed by these
papers:

- Long Phan et al., *Humanity's Last Exam*, arXiv:2501.14249 (2025). Expert-written,
  closed-form questions that are hard to look up.
- Xin Xu et al., *UGPhysics: A Comprehensive Benchmark for Undergraduate Physics Reasoning with
  Large Language Models*, arXiv:2502.00334 (2025). Rule-plus-model answer judging for physics.
- Xinyu Zhang et al., *PhysReason: A Comprehensive Benchmark towards Physics-Based Reasoning*,
  arXiv:2502.12054 (2025). Step-level scoring of physics solutions.
- Anisha Gunjal et al., *Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains*,
  arXiv:2507.17746 (2025). Rubric items as structured grading signals.
- Lianmin Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*,
  arXiv:2306.05685 (2023). Known biases of model judges, which is why the final answer is never
  left to the judge here.

## License

MIT © 2026 Satish, Siyaramji Tech Solutions. Problems are released under the same license.
They are new and were not copied from textbooks or existing benchmarks.
