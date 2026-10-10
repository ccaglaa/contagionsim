"""Fund and value trader tests"""

from __future__ import annotations

import math

import numpy as np
import pytest

from contagionsim.agents import (
    Fund,
    FundParams,
    FundShock,
    MarketMaker,
    NaiveLiquidation,
    NoiseTrader,
    ValueTrader,
    ValueTraderParams,
)
from contagionsim.market import Agent, Market, Simulation
from contagionsim.orderbook import Side

SHOCK_T = 3050  # after build-up (fund holds ~15% of float), not on a flow step


def fund_sim(seed: int, shocks: list[FundShock] | None = None, **fund_kw: float) -> Simulation:
    agents: list[Agent] = [NoiseTrader(f"n{i}") for i in range(20)]
    agents.append(MarketMaker("mm"))
    agents += [ValueTrader(f"v{i}") for i in range(5)]
    agents.append(Fund("fund", FundParams(**fund_kw), shocks))
    return Simulation(Market(initial_price=1000), agents, seed=seed)


def get_fund(sim: Simulation) -> Fund:
    fund = sim.agents[-1]
    assert isinstance(fund, Fund)
    return fund


# Accounting


def test_cash_is_conserved_including_external_flows() -> None:
    sim = fund_sim(seed=0, shocks=[FundShock(SHOCK_T, redeem_fraction=0.3)])
    for _ in range(SHOCK_T + 300):
        sim.step()
        assert sim.market.accounts["fund"].cash >= 0
    accounts = sim.market.accounts.values()
    assert sum(a.cash for a in accounts) == sim.market.external_cash
    assert sum(a.position for a in accounts) == 0


def test_flows_work_before_lookback_history_exists() -> None:
    # Regression: the first flow events have less history than flow_lookback
    sim = fund_sim(seed=1, flow_interval=10, flow_lookback=50)
    sim.run(200)
    assert len(get_fund(sim).records) == 200


def test_nav_per_unit_starts_at_one_try() -> None:
    sim = fund_sim(seed=0)
    sim.step()
    assert get_fund(sim).records[0].nav_per_unit == pytest.approx(100.0, rel=0.01)


# Limits and forced selling


def test_fund_never_exceeds_its_concentration_limit() -> None:
    sim = fund_sim(seed=2, max_float_share=0.05)  # 5,000 shares
    sim.run(3000)
    assert max(r.shares for r in get_fund(sim).records) <= 5000


def test_cap_forces_the_fund_below_the_cap() -> None:
    sim = fund_sim(seed=0, shocks=[FundShock(SHOCK_T, cap_fraction=0.02)])
    sim.run(SHOCK_T)
    assert get_fund(sim).records[-1].shares > 2000  # cap binds
    sim.run(200)
    fund = get_fund(sim)
    assert fund.records[-1].shares <= 2000
    assert sum(r.shares_sold for r in fund.records[SHOCK_T:]) > 0


def test_redemption_shock_is_paid_and_units_shrink() -> None:
    sim = fund_sim(seed=3, shocks=[FundShock(SHOCK_T, redeem_fraction=0.3)])
    sim.run(SHOCK_T)
    units_before = get_fund(sim).units
    sim.step()
    fund = get_fund(sim)
    assert fund.units == pytest.approx(0.7 * units_before, rel=0.05)  # flows may add a little
    sim.run(200)
    assert fund.pending == 0.0
    assert fund.failed_at is None


def test_redeemers_are_paid_at_pre_sale_nav_and_stayers_bear_the_cost() -> None:
    # First-mover advantage, in a tiny deterministic market
    m = Market(1000)
    m.register("other")
    fund = Fund("fund", FundParams(), [FundShock(1, redeem_fraction=0.99)])
    m.register("fund")
    rng = np.random.default_rng(0)
    m.submit_limit("other", Side.SELL, 10_000, 1001)
    fund.act(0, m, rng)  # launch (50,000 units at 1 TRY) and buy 100 shares at 10.01
    assert m.accounts["fund"].position == 100

    m.submit_limit("other", Side.BUY, 1_000, 900)  # the only bid is far below
    nav_before = fund.nav_per_unit(m)
    owed = 0.99 * fund.units * nav_before

    fund.act(1, m, rng)  # redemption: cash is not enough, it must sell at 9.00
    paid = fund.params.initial_cash - m.external_cash  # launch cash minus payouts
    assert paid + fund.pending == pytest.approx(owed, rel=1e-9)  # claim fixed before selling
    assert fund.records[-1].shares_sold > 0
    assert fund.nav_per_unit(m) < nav_before  # the fire-sale loss lands on the stayers


def test_naive_liquidation_sells_everything_now() -> None:
    assert NaiveLiquidation().shares_to_sell(1234, Market(1000)) == 1234


# Scenario: the shock makes the crash


def test_shock_creates_a_fire_sale_discount() -> None:
    shocks = [FundShock(SHOCK_T, cap_fraction=0.08), FundShock(SHOCK_T, redeem_fraction=0.3)]
    calm = fund_sim(seed=0).run(SHOCK_T + 500)
    shocked = fund_sim(seed=0, shocks=shocks).run(SHOCK_T + 500)
    assert calm[:SHOCK_T] == shocked[:SHOCK_T]  # identical until the shock
    low_calm = min(s.last_price for s in calm[SHOCK_T:])
    low_shocked = min(s.last_price for s in shocked[SHOCK_T:])
    assert low_shocked < low_calm * 0.95


# Value trader


def value_market(ref: int) -> tuple[Market, ValueTrader]:
    m = Market(ref)
    vt = ValueTrader("v", ValueTraderParams(fundamental=1000, p_act=1.0, max_short=100))
    m.register("v")
    return m, vt


def test_value_trader_buys_when_underpriced() -> None:
    m, vt = value_market(800)  # 20% below value
    vt.act(0, m, np.random.default_rng(0))
    assert m.book.best_bid() == 800
    assert m.book.depth(Side.BUY, 1)[0][1] == 400  # 20 shares per pct x 20 pct


def test_value_trader_respects_short_limit() -> None:
    m, vt = value_market(1500)  # 50% overpriced: wants 1000, may short only 100
    vt.act(0, m, np.random.default_rng(0))
    assert m.book.depth(Side.SELL, 1) == [(1500, 100)]


def test_value_trader_idle_inside_threshold() -> None:
    m, vt = value_market(1030)  # 3% < 5% threshold
    vt.act(0, m, np.random.default_rng(0))
    assert len(m.book) == 0


def test_value_trader_at_long_limit_does_nothing() -> None:
    m, vt = value_market(500)
    m.accounts["v"].position = 5000
    vt.act(0, m, np.random.default_rng(0))
    assert len(m.book) == 0
    assert math.isfinite(m.reference_price())
