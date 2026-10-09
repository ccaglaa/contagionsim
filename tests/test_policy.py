"""Policy levers (Q3): redemption gate and regulatory cap phase-in"""

from __future__ import annotations

import numpy as np
import pytest

from contagionsim.agents import FundShock, PortfolioFund, PortfolioFundParams
from contagionsim.execution import Immediate, Linear, Schedule
from contagionsim.market import ANA, YILDIZ, Exchange, Listing
from contagionsim.orderbook import Side

LIQ = ("L1", "L2", "L3", "L4")


def exchange_with_deep_bids() -> Exchange:
    """Fund holds 1,000 shares of each of 5 stocks at 10.00 TRY: NAV = 5,000,000 ticks"""
    listings = [Listing("CORNER", 1000, 200_000, ANA)]
    listings += [Listing(s, 1000, 2_000_000, YILDIZ) for s in LIQ]
    ex = Exchange(listings)
    ex.register("mm", cash=10**12)
    ex.register("fund", positions={"CORNER": 1000, **{s: 1000 for s in LIQ}})
    ex.begin_step(0)
    for s in ("CORNER", *LIQ):
        ex[s].submit_limit("mm", Side.BUY, 100_000, 1000)
        ex[s].last_price = 1000
    return ex


def make_fund(
    ex: Exchange,
    shocks: list[FundShock],
    gate: float | None = None,
    cap_schedule: Schedule | None = None,
) -> PortfolioFund:
    params = PortfolioFundParams(initial_cash=0, redemption_gate=gate)
    fund = PortfolioFund("fund", params, shocks, Immediate(), cap_schedule)
    fund.act(0, ex, np.random.default_rng(0))
    fund.units = 50_000.0  # 100 ticks (1 TRY) per unit
    return fund


def test_gate_queues_redemptions_above_the_daily_allowance() -> None:
    ex = exchange_with_deep_bids()
    fund = make_fund(ex, [FundShock(1, redeem_fraction=0.5)], gate=0.2)
    nav_per_unit = fund.nav_per_unit(ex)
    fund.act(1, ex, np.random.default_rng(0))
    # 50% of 5,000,000 is redeemed; only 20% of the fund (1,000,000) is payable today
    assert fund.gated == pytest.approx(1_500_000, rel=1e-6)
    assert fund.pending == pytest.approx(0.0, abs=1.0)  # today's tranche already paid
    # The queue is still owed, so the stayers' NAV per unit is not flattered
    assert fund.nav_per_unit(ex) == pytest.approx(nav_per_unit, rel=1e-6)


def test_gate_releases_a_new_tranche_each_day() -> None:
    ex = exchange_with_deep_bids()
    fund = make_fund(ex, [FundShock(1, redeem_fraction=0.5)], gate=0.2)
    rng = np.random.default_rng(0)
    fund.act(1, ex, rng)
    # Steps 550 and 1001 avoid the investor-flow steps (every 100), which would add flows
    fund.act(550, ex, rng)  # same day: no new allowance
    assert fund.gated == pytest.approx(1_500_000, rel=1e-6)
    fund.act(1001, ex, rng)  # next day: 20% of the fund's 4,000,000 gross assets
    assert fund.gated == pytest.approx(700_000, rel=1e-6)


def test_no_gate_pays_everything_at_once() -> None:
    ex = exchange_with_deep_bids()
    fund = make_fund(ex, [FundShock(1, redeem_fraction=0.5)])
    fund.act(1, ex, np.random.default_rng(0))
    assert fund.gated == 0.0 and fund.pending == pytest.approx(0.0, abs=1.0)


def test_cap_phase_in_is_separate_from_redemption_schedule() -> None:
    ex = exchange_with_deep_bids()
    shocks = [FundShock(1, cap_fraction=0.002)]  # cap 200 shares: 800 excess
    fund = make_fund(ex, shocks, cap_schedule=Linear(4))
    rng = np.random.default_rng(0)
    held = []
    for t in range(1, 5):
        fund.act(t, ex, rng)
        held.append(ex.position("fund", "CORNER"))
    assert held == [800, 600, 400, 200]
