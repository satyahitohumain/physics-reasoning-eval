import pytest

pytest.importorskip("scipy")

from verification.checks import CHECKS  # noqa: E402


@pytest.mark.parametrize("pid", sorted(CHECKS))
def test_reference_matches_independent_numerics(pid):
    fn, tol = CHECKS[pid]
    got, ref = fn()
    assert abs(got - ref) / abs(ref) <= tol, (got, ref)
