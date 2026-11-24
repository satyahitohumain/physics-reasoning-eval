"""Independent numerical checks of every reference answer.

Each check recomputes the answer by a route that does not reuse the reference derivation
(direct simulation, a different expansion, or brute-force diagonalisation) and compares it
with the closed form stored in the problem file. A reference answer that fails here is a bug
in the dataset, so these run in CI.

Run directly:  python -m verification.checks
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq, minimize_scalar

from physeval.problems import Problem, load_all

PROBLEMS = load_all()


def reference(pid: str, **values: float) -> float:
    """Evaluate a problem's reference answer at the given symbol values."""
    p: Problem = PROBLEMS[pid]
    if p.answer.type == "numeric":
        return float(p.answer.value)
    subs = {p.answer.symbols[k]: v for k, v in values.items()}
    return float(p.answer.expr.subs(subs))


def _period_from_crossings(t: np.ndarray, y: np.ndarray) -> float:
    """Mean period from upward zero crossings (linear interpolation)."""
    idx = np.where((y[:-1] < 0) & (y[1:] >= 0))[0]
    times = t[idx] - y[idx] * (t[idx + 1] - t[idx]) / (y[idx + 1] - y[idx])
    if len(times) < 3:
        raise RuntimeError("not enough oscillations to measure a period")
    return float(np.mean(np.diff(times)))


# --------------------------------------------------------------------------- PHY-001
def check_phy001() -> tuple[float, float]:
    """Simulate the full nonlinear equation of motion for a tiny displacement."""
    w, g, R = 7.0, 9.81, 0.5
    th0 = math.acos(g / (R * w * w))

    def rhs(_, y):
        th, om = y
        return [om, w * w * math.sin(th) * math.cos(th) - (g / R) * math.sin(th)]

    sol = solve_ivp(rhs, (0, 20), [th0 + 1e-4, 0.0], max_step=1e-3, rtol=1e-11, atol=1e-13)
    measured = 2 * math.pi / _period_from_crossings(sol.t, sol.y[0] - th0)
    return measured, reference("PHY-001", w=w, g=g, R=R)


# --------------------------------------------------------------------------- PHY-002
def check_phy002() -> tuple[float, float]:
    """Minimise the energy functional by quadrature; also confirm it bounds the exact energy."""
    F, hbar, m = 1.3, 0.7, 2.1
    x = np.linspace(-60, 60, 120001)
    dx = x[1] - x[0]

    def energy(sigma: float) -> float:
        psi = np.exp(-x**2 / (4 * sigma**2))
        norm = np.sum(psi**2) * dx
        dpsi = np.gradient(psi, dx)
        kinetic = hbar**2 / (2 * m) * np.sum(dpsi**2) * dx / norm
        potential = np.sum(F * np.abs(x) * psi**2) * dx / norm
        return kinetic + potential

    best = minimize_scalar(energy, bounds=(0.05, 10), method="bounded", options={"xatol": 1e-10}).fun

    # Exact ground state by finite differences, to confirm the variational inequality.
    n, L = 6000, 25.0
    xs = np.linspace(-L, L, n)
    h = xs[1] - xs[0]
    diag = hbar**2 / (m * h * h) + F * np.abs(xs)
    off = -hbar**2 / (2 * m * h * h) * np.ones(n - 1)
    exact = eigh_tridiagonal(diag, off, select="i", select_range=(0, 0))[0][0]
    airy = 1.0187929716 * (hbar**2 * F**2 / (2 * m)) ** (1 / 3)
    assert abs(exact - airy) / airy < 1e-4, (exact, airy)
    assert best > exact
    return best, reference("PHY-002", F=F, hbar=hbar, m=m)


# --------------------------------------------------------------------------- PHY-003
def check_phy003() -> tuple[float, float]:
    """Differentiate the internal energy numerically and locate the heat-capacity peak."""

    def U(t: float) -> float:  # energy per system, Delta = k_B = 1
        b = 1 / t
        return 3 * math.exp(-b) / (1 + 3 * math.exp(-b))

    def minus_C(t: float, h: float = 1e-5) -> float:
        return -(U(t + h) - U(t - h)) / (2 * h)

    peak = minimize_scalar(minus_C, bounds=(0.1, 1.5), method="bounded", options={"xatol": 1e-9}).x
    return peak, reference("PHY-003")


# --------------------------------------------------------------------------- PHY-004
def check_phy004() -> tuple[float, float]:
    """Sum the multipole (Legendre) series of the induced surface charge instead of using images.

    Outside a neutral conducting sphere in the field of an external charge, the induced
    potential is sum_{l>=1} B_l P_l(cos th) / r^(l+1) with B_l = -k q a^(2l+1) / d^(l+1)
    (the l = 0 term vanishes because the sphere is neutral). The radial field at the charge
    is sum (l+1) B_l / d^(l+2).
    """
    q, a, d, eps0 = 1.7, 0.8, 1.3, 1.1
    k = 1 / (4 * math.pi * eps0)
    force = 0.0
    for l in range(1, 4000):
        term = (l + 1) * k * q * q * a ** (2 * l + 1) / d ** (2 * l + 3)
        force += term
        if term < 1e-18 * force:
            break
    # Far-field check: polarisability of a conducting sphere gives 2 k q^2 a^3 / d^5.
    far = reference("PHY-004", q=1.0, a=1e-2, d=1.0, epsilon_0=1.0)
    dipole = 2 * 1e-2**3 / (4 * math.pi)
    assert abs(far / dipole - 1) < 1e-3, (far, dipole)
    return force, reference("PHY-004", q=q, a=a, d=d, epsilon_0=eps0)


# --------------------------------------------------------------------------- PHY-005
def check_phy005() -> tuple[float, float]:
    """Integrate the Kepler orbit after the kick and measure r_max - r_min."""
    G, M, r0, dv = 1.0, 1.0, 1.0, 1e-4
    v0 = math.sqrt(G * M / r0)

    def rhs(_, y):
        x, yy, vx, vy = y
        r3 = (x * x + yy * yy) ** 1.5
        return [vx, vy, -G * M * x / r3, -G * M * yy / r3]

    period = 2 * math.pi * math.sqrt(r0**3 / (G * M))
    sol = solve_ivp(rhs, (0, 1.2 * period), [r0, 0, dv, v0], rtol=1e-12, atol=1e-14, max_step=period / 4000)
    r = np.hypot(sol.y[0], sol.y[1])
    return float(r.max() - r.min()), reference("PHY-005", dv=dv, r0=r0, G=G, M=M)


# --------------------------------------------------------------------------- PHY-006
def check_phy006() -> tuple[float, float]:
    """Derive the equations of motion from the full two-coordinate Lagrangian and simulate."""
    R_, r_, M_, m_, g_ = 1.0, 0.3, 2.0, 1.5, 9.81
    t = sp.Symbol("t")
    phi, psi = sp.Function("phi")(t), sp.Function("psi")(t)
    a = R_ - r_
    chi_dot = (a * phi.diff(t) - R_ * psi.diff(t)) / r_  # rolling constraint
    T = (
        sp.Rational(1, 2) * m_ * a**2 * phi.diff(t) ** 2
        + sp.Rational(1, 2) * (sp.Rational(1, 2) * m_ * r_**2) * chi_dot**2
        + sp.Rational(1, 2) * M_ * R_**2 * psi.diff(t) ** 2
    )
    V = -m_ * g_ * a * sp.cos(phi)
    Lag = T - V
    eqs = [sp.diff(Lag.diff(q.diff(t)), t) - Lag.diff(q) for q in (phi, psi)]
    acc = sp.solve(eqs, [phi.diff(t, 2), psi.diff(t, 2)], dict=True)[0]
    p, s, pd, sd = sp.symbols("p s pd sd")
    rep = {phi.diff(t): pd, psi.diff(t): sd, phi: p, psi: s}
    f_phi = sp.lambdify((p, s, pd, sd), acc[phi.diff(t, 2)].subs(rep))
    f_psi = sp.lambdify((p, s, pd, sd), acc[psi.diff(t, 2)].subs(rep))

    def rhs(_, y):
        return [y[2], y[3], f_phi(*y), f_psi(*y)]

    sol = solve_ivp(rhs, (0, 20), [1e-4, 0, 0, 0], rtol=1e-11, atol=1e-14, max_step=1e-3)
    measured = _period_from_crossings(sol.t, sol.y[0])
    return measured, reference("PHY-006", R=R_, r=r_, M=M_, m=m_, g=g_)


# --------------------------------------------------------------------------- PHY-007
def check_phy007() -> tuple[float, float]:
    """Solve energy and momentum conservation numerically for the scattered photon energy."""
    E0, m, c, beta = 0.8, 1.0, 1.0, 0.6
    gamma = 1 / math.sqrt(1 - beta**2)
    E_tot = gamma * m * c**2 + E0
    p_tot = gamma * m * beta * c - E0 / c  # photon initially moves in -x

    def mismatch(E1: float) -> float:
        p_e = p_tot - E1 / c  # scattered photon moves in +x
        E_e = E_tot - E1
        return E_e**2 - (p_e * c) ** 2 - (m * c**2) ** 2

    E1 = brentq(mismatch, 1e-9, E_tot * (1 - 1e-12))
    return E1, reference("PHY-007", E0=E0, m=m, c=c, beta=beta)


# --------------------------------------------------------------------------- PHY-008
def check_phy008() -> tuple[float, float]:
    """Diagonalise the perturbed well and extract the O(V0^2) coefficient by a central difference."""
    m, a, hbar = 1.0, 1.0, 1.0
    n = 4000  # interior grid points; even, so the step sits exactly between two nodes
    h = a / (n + 1)
    x = h * np.arange(1, n + 1)
    kin = hbar**2 / (2 * m * h * h)

    def ground(V0: float) -> float:
        diag = 2 * kin + V0 * (x < a / 2)
        off = -kin * np.ones(n - 1)
        return eigh_tridiagonal(diag, off, select="i", select_range=(0, 0))[0][0]

    lam = 0.2
    coeff = (ground(lam) + ground(-lam) - 2 * ground(0.0)) / (2 * lam**2)
    return coeff, reference("PHY-008", m=m, a=a, V0=1.0, hbar=hbar)


CHECKS: dict[str, tuple[Callable[[], tuple[float, float]], float]] = {
    "PHY-001": (check_phy001, 1e-4),
    "PHY-002": (check_phy002, 1e-5),
    "PHY-003": (check_phy003, 1e-5),
    "PHY-004": (check_phy004, 1e-9),
    "PHY-005": (check_phy005, 1e-3),
    "PHY-006": (check_phy006, 1e-4),
    "PHY-007": (check_phy007, 1e-10),
    "PHY-008": (check_phy008, 1e-3),
}


def main() -> int:
    bad = 0
    for pid, (fn, tol) in CHECKS.items():
        got, ref = fn()
        err = abs(got - ref) / abs(ref)
        ok = err <= tol
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {pid}  numerical={got:.10g}  reference={ref:.10g}  rel.err={err:.1e}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
