import numpy as np
import pytest
from scipy.stats import beta

from priomed_classification.np_threshold import InfeasibleError, min_calibration_size, np_threshold


def test_infeasible_when_too_few_high_cases():
    with pytest.raises(InfeasibleError):
        np_threshold(np.random.rand(10), alpha=0.05, delta=0.05)


def test_min_size_formula():
    # (1-0.05)^59 ≈ 0.0485 <= 0.05 ; (0.95)^58 ≈ 0.051 > 0.05
    assert min_calibration_size(0.05, 0.05) == 59


def test_empirical_coverage_of_np_guarantee():
    """P(tasa real de omisión > alpha) debe ser <= delta (con tolerancia de simulación)."""
    rng = np.random.default_rng(0)
    alpha, delta, n, trials = 0.05, 0.05, 200, 2000
    dist = beta(5, 2)  # puntajes de casos HIGH: continuos, cualquier distribución sirve
    violations = 0
    for _ in range(trials):
        t = np_threshold(dist.rvs(size=n, random_state=rng), alpha, delta)
        true_miss = dist.cdf(t)  # P(score < t | HIGH)
        violations += true_miss > alpha
    rate = violations / trials
    assert rate <= delta + 0.02, f"tasa de violación {rate:.3f} excede delta={delta}"
