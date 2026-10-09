"""VaR models and backtests, checked against known answers"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from contagionsim.analysis.var import (
    Garch11,
    backtest,
    christoffersen_ind,
    fit_garch11,
    historical_var,
    hits,
    kupiec_pof,
    normal_var,
    transition_counts,
)


def simulate_garch(params: Garch11, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    s2 = params.long_run_variance
    for t in range(n):
        r[t] = params.mu + np.sqrt(s2) * rng.standard_normal()
        e = r[t] - params.mu
        s2 = params.omega + params.alpha * e * e + params.beta * s2
    return r


def test_garch_fit_recovers_known_parameters() -> None:
    true = Garch11(mu=0.0, omega=2e-6, alpha=0.10, beta=0.85)  # persistence 0.95
    fitted = fit_garch11(simulate_garch(true, 6000, seed=1))
    assert fitted.alpha == pytest.approx(0.10, abs=0.03)
    assert fitted.beta == pytest.approx(0.85, abs=0.05)
    assert fitted.long_run_variance == pytest.approx(true.long_run_variance, rel=0.25)


def test_garch_variance_reacts_to_a_shock() -> None:
    g = Garch11(mu=0.0, omega=1e-6, alpha=0.1, beta=0.85)
    calm = g.variances(np.zeros(5))
    shocked = g.variances(np.array([0.0, 0.0, -0.05, 0.0, 0.0]))
    assert shocked[3] > 10 * calm[3]  # big variance right after the shock
    assert shocked[4] < shocked[3]  # then it decays


def test_garch_fit_without_clustering_keeps_a_sensible_variance() -> None:
    # Regression: on iid data the old fit could return alpha ~ 0, beta ~ 1, omega ~ 0,
    # whose long-run variance is near zero, so its VaR was far too small
    r = np.random.default_rng(8).normal(0, 0.003, 160)
    g = fit_garch11(r)
    assert g.long_run_variance == pytest.approx(np.var(r), rel=1e-6)  # variance targeting
    var = g.var(r)
    assert var.min() > 0.5 * (-stats.norm.ppf(0.01) * r.std())


def test_garch_fit_needs_enough_data() -> None:
    with pytest.raises(ValueError):
        fit_garch11(np.zeros(10) + 0.01)


def test_rolling_models_use_only_the_past() -> None:
    r = np.r_[np.zeros(50), -1.0]  # a crash in the last period
    for model in (historical_var, normal_var):
        v = model(r, window=50)
        assert np.isnan(v[:50]).all()
        assert v[50] == pytest.approx(0.0, abs=1e-12)  # the crash is not in its window


def test_normal_var_matches_the_formula() -> None:
    r = np.random.default_rng(0).normal(0.001, 0.02, 300)
    v = normal_var(r, window=250, level=0.99)
    past = r[49:299]  # the 250 returns before t = 299
    expected = -(past.mean() + stats.norm.ppf(0.01) * past.std(ddof=1))
    assert v[299] == pytest.approx(expected)


def test_kupiec_accepts_the_expected_count_and_rejects_too_many() -> None:
    lr, p = kupiec_pof(1000, 10, 0.01)  # exactly the expected 1%
    assert lr == pytest.approx(0.0, abs=1e-9) and p == pytest.approx(1.0)
    _, p_bad = kupiec_pof(1000, 30, 0.01)  # 3x too many
    assert p_bad < 0.001
    _, p_zero = kupiec_pof(1000, 0, 0.01)  # zero exceptions is also suspicious over 1000 days
    assert p_zero < 0.01


def test_transition_counts() -> None:
    h = np.array([0, 1, 1, 0, 0, 1], dtype=bool)
    assert transition_counts(h) == (1, 2, 1, 1)  # 00, 01 x2, 10, 11


def test_christoffersen_detects_clustering() -> None:
    scattered = np.zeros(500, dtype=bool)
    scattered[::50] = True  # 10 isolated hits
    clustered = np.zeros(500, dtype=bool)
    clustered[200:210] = True  # 10 hits in a row
    _, p_scattered = christoffersen_ind(*transition_counts(scattered))
    _, p_clustered = christoffersen_ind(*transition_counts(clustered))
    assert p_scattered > 0.05
    assert p_clustered < 0.001


def test_backtest_pools_series_without_joining_them() -> None:
    a = np.array([0, 0, 1], dtype=bool)
    b = np.array([1, 0, 0], dtype=bool)
    bt = backtest([a, b], level=0.99)
    assert (bt.n, bt.exceptions) == (6, 2)
    # a's last hit and b's first hit must NOT count as a hit-hit transition
    assert transition_counts(np.r_[a, b])[3] == 1
    pooled = np.sum([transition_counts(a), transition_counts(b)], axis=0)
    assert pooled[3] == 0


def test_well_specified_model_passes_on_normal_data() -> None:
    r = np.random.default_rng(3).normal(0, 0.01, 5000)
    v = np.full(len(r), -stats.norm.ppf(0.01) * 0.01)
    bt = backtest([hits(r, v)], level=0.99)
    assert not bt.rejected()
