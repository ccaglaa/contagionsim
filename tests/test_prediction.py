"""Q4 early warning: scenarios, labels, sampling and, above all, no look-ahead"""

from __future__ import annotations

import numpy as np
import pytest

from contagionsim.prediction import (
    FEATURES,
    FLOW_FEATURES,
    draw_scenario,
    fund_flow_features,
    run_features,
    run_labels,
    sample_steps,
    simulate,
)
from contagionsim.prediction.runs import RunScenario
from contagionsim.scenarios import MiniBistConfig

LIQ = ("L1", "L2", "L3", "L4")
LIMIT = MiniBistConfig().concentrated_rules.daily_limit


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def test_scenarios_are_reproducible_and_varied() -> None:
    assert draw_scenario(5) == draw_scenario(5)
    draws = [draw_scenario(s) for s in range(300)]
    calm = sum(d.is_calm for d in draws) / len(draws)
    assert 0.2 < calm < 0.4  # p_calm = 0.3
    runs = [d for d in draws if not d.is_calm]
    assert all(1 <= len(d.wave_times) <= 6 for d in runs)
    assert all(0.05 <= f <= 0.20 for d in runs for f in d.wave_sizes)
    assert all(7050 <= d.wave_times[0] < 8800 for d in runs)


def test_scenario_shocks() -> None:
    sc = RunScenario(0, (7100, 7200), (0.1, 0.05), cap=True, twap=False, days=10)
    shocks = sc.shocks()
    assert [(s.t, s.redeem_fraction) for s in shocks[:2]] == [(7100, 0.1), (7200, 0.05)]
    assert shocks[2].t == 7100 and shocks[2].cap_fraction == 0.08


# ---------------------------------------------------------------------------
# Labels and sampling
# ---------------------------------------------------------------------------


def test_labels_mark_the_horizon_before_each_breaker() -> None:
    y = run_labels(20, breaker_steps=[10], horizon=3)
    assert list(np.flatnonzero(y)) == [7, 8, 9]  # breaker in (t, t + 3] for t = 7, 8, 9
    assert not y[10]  # at the breaker step itself the event is no longer in the future


@pytest.fixture(scope="module")
def short_run():  # type: ignore[no-untyped-def]
    # Waves at 1250 and 1350: off the investor-flow steps (every 100), so they are not mixed
    sc = RunScenario(3, (1250, 1350), (0.15, 0.15), cap=True, twap=False, days=2)
    return simulate(sc)


def test_features_have_no_look_ahead(short_run) -> None:  # type: ignore[no-untyped-def]
    # The decisive test: features at step t must not change when the future is removed
    sim, fund = short_run
    h = sim.history
    halts = [x for m in sim.exchange.markets.values() for x in m.halts]
    full = run_features(h, halts, "CORNER", LIQ, LIMIT)
    for t in (300, 999, 1000, 1250, 1500):
        past_halts = [x for x in halts if x <= t]
        truncated = run_features(h[: t + 1], past_halts, "CORNER", LIQ, LIMIT)
        np.testing.assert_allclose(truncated[t], full[t], err_msg=f"look-ahead at step {t}")


def test_features_shape_and_finite(short_run) -> None:  # type: ignore[no-untyped-def]
    sim, _ = short_run
    halts = [x for m in sim.exchange.markets.values() for x in m.halts]
    X = run_features(sim.history, halts, "CORNER", LIQ, LIMIT)
    assert X.shape == (len(sim.history), len(FEATURES))
    assert np.isfinite(X).all()


def test_sampling_skips_halts_and_the_rest_of_a_breaker_day(short_run) -> None:  # type: ignore[no-untyped-def]
    sim, _ = short_run
    h = sim.history
    steps = sample_steps(h, breaker_steps=[1400], start=0, every=5)
    assert all(not h[s].index_halted for s in steps)
    assert not any(1400 <= s < 2000 for s in steps)  # 5% stage already used that day
    assert all(s % 5 == 0 for s in steps)


def test_fund_flow_features_see_the_redemption_wave(short_run) -> None:  # type: ignore[no-untyped-def]
    # Wave of 15% of units at step 1250: the regulator view must show it at once
    _, fund = short_run
    F = fund_flow_features(fund)
    assert F.shape == (len(fund.records), len(FLOW_FEATURES))
    assert np.isfinite(F).all()
    out_today = F[:, FLOW_FEATURES.index("fund_net_outflow_today")]
    units = fund.units_history
    # The jump is 15% of the units outstanding just before, measured against units at the open
    expected_jump = 0.15 * units[1249] / units[999]
    assert out_today[1250] - out_today[1249] == pytest.approx(expected_jump, rel=1e-6)
    assert F[1250, FLOW_FEATURES.index("fund_sold_today")] > 0.05  # it started selling
