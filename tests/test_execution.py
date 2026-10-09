"""Liquidation schedules (Q2), the fund following them, and the news shock"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from contagionsim.agents import FundShock, PortfolioFund, PortfolioFundParams
from contagionsim.execution import AlmgrenChriss, Immediate, Linear, kappa_from_model
from contagionsim.market import ANA, YILDIZ, Exchange, Listing
from contagionsim.orderbook import Side
from contagionsim.scenarios import MiniBistConfig, News, mini_bist

LIQ = ("L1", "L2", "L3", "L4")

# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def test_immediate_is_always_done() -> None:
    assert Immediate().fraction_done(0) == 1.0


def test_linear_is_a_straight_line_reaching_one_at_the_horizon() -> None:
    s = Linear(10)
    assert [s.fraction_done(e) for e in (0, 4, 9, 50)] == [0.1, 0.5, 1.0, 1.0]


def test_almgren_chriss_endpoints_and_monotone() -> None:
    s = AlmgrenChriss(100, 5.0)
    f = [s.fraction_done(e) for e in range(100)]
    assert f[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(f, f[1:], strict=False))


def test_almgren_chriss_tends_to_twap_when_patient() -> None:
    ac, twap = AlmgrenChriss(100, 1e-9), Linear(100)
    assert all(ac.fraction_done(e) == pytest.approx(twap.fraction_done(e)) for e in range(100))


def test_urgency_front_loads_the_selling() -> None:
    early = 9  # after 10% of the horizon
    patient = AlmgrenChriss(100, 1.0).fraction_done(early)
    urgent = AlmgrenChriss(100, 10.0).fraction_done(early)
    assert Linear(100).fraction_done(early) < patient < urgent
    # k = 10: about 63% done after 10% of the time, 1 - sinh(9) / sinh(10)
    assert urgent == pytest.approx(1 - math.sinh(9) / math.sinh(10))


@pytest.mark.parametrize("bad", [lambda: Linear(0), lambda: AlmgrenChriss(0, 1.0),
                                 lambda: AlmgrenChriss(10, -1.0)])  # fmt: skip
def test_invalid_schedules_raise(bad) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        bad()


def test_kappa_from_model() -> None:
    assert kappa_from_model(4.0, 0.5, 1.0) == pytest.approx(1.0)  # sqrt(4 * 0.25 / 1)
    assert kappa_from_model(0.0, 0.5, 1.0) == 0.0  # no risk aversion: TWAP
    with pytest.raises(ValueError):
        kappa_from_model(1.0, 0.1, 0.0)


# ---------------------------------------------------------------------------
# The fund following a schedule
# ---------------------------------------------------------------------------


def exchange_with_deep_bids() -> Exchange:
    listings = [Listing("CORNER", 1000, 200_000, ANA)]
    listings += [Listing(s, 1000, 2_000_000, YILDIZ) for s in LIQ]
    ex = Exchange(listings)
    ex.register("mm", cash=10**12)
    ex.register("fund", positions={"CORNER": 1000, **{s: 1000 for s in LIQ}})
    ex.begin_step(0)
    for s in ("CORNER", *LIQ):
        ex[s].submit_limit("mm", Side.BUY, 100_000, 999)
    return ex


def fund_on(ex: Exchange, schedule: object, shocks: list[FundShock]) -> PortfolioFund:
    fund = PortfolioFund("fund", PortfolioFundParams(initial_cash=0), shocks, schedule)  # type: ignore[arg-type]
    fund.act(0, ex, np.random.default_rng(0))  # initialize: no cash, nothing to invest
    fund.units = 50_000.0  # NAV 5,000,000 ticks -> 100 ticks (1 TRY) per unit
    return fund


def test_linear_schedule_pays_redeemers_gradually() -> None:
    ex = exchange_with_deep_bids()
    fund = fund_on(ex, Linear(10), [FundShock(1, redeem_fraction=0.10)])
    rng = np.random.default_rng(0)
    fund.act(1, ex, rng)
    owed = 0.10 * 50_000 * 100  # 10% of 50,000 units at 1 TRY (100 ticks) each
    assert fund.pending == pytest.approx(0.9 * owed, rel=0.02)  # 1/10 paid on the first step
    for t in range(2, 11):
        fund.act(t, ex, rng)
    assert fund.pending < 0.01 * owed  # done by the horizon


def test_immediate_schedule_pays_at_once() -> None:
    ex = exchange_with_deep_bids()
    fund = fund_on(ex, Immediate(), [FundShock(1, redeem_fraction=0.10)])
    fund.act(1, ex, np.random.default_rng(0))
    assert fund.pending < 0.01 * 0.10 * 50_000 * 100


def test_cap_excess_is_sold_on_schedule() -> None:
    ex = exchange_with_deep_bids()
    fund = fund_on(ex, Linear(4), [FundShock(1, cap_fraction=0.002)])  # cap: 200 shares
    rng = np.random.default_rng(0)
    held = []
    for t in range(1, 5):
        fund.act(t, ex, rng)
        held.append(ex.position("fund", "CORNER"))
    assert held == [800, 600, 400, 200]  # 800 excess shares, a quarter per step


def test_schedule_never_changes_runs_without_a_shock() -> None:
    # Without stress there is nothing to liquidate: the schedule must be irrelevant
    twap = dataclasses.replace(MiniBistConfig(), schedule=Linear(1000))
    a, _ = mini_bist(seed=4)
    b, _ = mini_bist(seed=4, config=twap)
    assert a.run(1500) == b.run(1500)


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------


def test_bad_news_moves_the_value_traders_fundamental() -> None:
    sim, _ = mini_bist(seed=0, news=[News(5, "CORNER", 0.8)])
    value_traders = [
        a.agent for a in sim.agents if getattr(a, "name", "").startswith("value_CORNER")
    ]  # type: ignore[attr-defined]
    sim.run(5)
    assert all(vt.params.fundamental == 1000 for vt in value_traders)
    sim.run(1)
    assert all(vt.params.fundamental == 800 for vt in value_traders)
