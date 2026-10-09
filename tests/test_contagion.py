"""M4 part 2: PortfolioFund behaviour and the contagion scenario"""

from __future__ import annotations

import numpy as np
import pytest

from contagionsim.agents import FundShock, PortfolioFund, PortfolioFundParams
from contagionsim.market import ANA, YILDIZ, Exchange, Listing
from contagionsim.orderbook import Side
from contagionsim.scenarios import MiniBistConfig, mini_bist

LIQ = ("L1", "L2", "L3", "L4")

# Small hand-built exchange for unit tests

def small_exchange() -> Exchange:
    listings = [Listing("CORNER", 1000, 200_000, ANA)]
    listings += [Listing(s, 1000, 2_000_000, YILDIZ) for s in LIQ]
    ex = Exchange(listings)
    ex.register("mm", cash=10**12)
    ex.begin_step(0)
    return ex


def fund_with_holdings(ex: Exchange) -> PortfolioFund:
    fund = PortfolioFund("fund", PortfolioFundParams(initial_cash=0))
    ex.register("fund", positions={"CORNER": 1000, **{s: 1000 for s in LIQ}})
    fund.act(0, ex, np.random.default_rng(0))  # initialize (no cash: nothing to invest)
    fund.units = 50_000.0
    return fund


def test_sellable_skips_halted_and_limit_down_stocks() -> None:
    ex = small_exchange()
    fund = fund_with_holdings(ex)
    assert fund.sellable(ex) == ["CORNER", *LIQ]
    ex["L1"].start_auction(10)  # halted
    corner = ex["CORNER"]
    corner.last_price = corner.floor  # limit-down, no bids
    assert fund.sellable(ex) == ["L2", "L3", "L4"]


def test_redemption_cash_comes_from_liquid_stocks_when_corner_is_locked() -> None:
    # The contagion channel in one step
    ex = small_exchange()
    fund = fund_with_holdings(ex)
    corner = ex["CORNER"]
    corner.last_price = corner.floor
    for s in LIQ:
        ex[s].submit_limit("mm", Side.BUY, 5000, 995)
    fund.pending = 400 * 1000.0  # owes 4,000 TRY
    fund.act(1, ex, np.random.default_rng(0))
    sold = fund.records[-1].sold
    assert "CORNER" not in sold
    assert set(sold) == set(LIQ)
    # Sold at 9.95 while estimating with 10.00: a small shortfall is retried next step
    assert fund.pending < 0.01 * 400 * 1000


def test_soft_close_blocks_subscriptions() -> None:
    # Cap well below the launch size, so trading costs cannot dip NAV under it
    cfg = MiniBistConfig(fund=PortfolioFundParams(initial_cash=30_000_000, max_aum=25_000_000))
    sim, fund = mini_bist(seed=0, config=cfg)
    sim.run(1500)
    assert all(r.flow <= 0 for r in fund.records)


# Scenario level

def test_fund_invests_in_both_and_respects_its_limit() -> None:
    sim, fund = mini_bist(seed=1)
    sim.run(3000)
    last = fund.records[-1]
    limit = fund.params.max_float_share * fund.params.concentrated_free_float
    assert 0 < last.positions["CORNER"] <= limit
    assert all(last.positions[s] > 0 for s in LIQ)
    assert last.cash < 0.2 * fund.nav(sim.exchange)  # invested, not sitting on cash (M3 flaw)
    ledger = sim.exchange.ledger
    assert sum(ledger.cash.values()) == ledger.external_cash


@pytest.mark.slow
def test_fire_sale_spreads_to_liquid_stocks_and_trips_index_breaker() -> None:
    T = 8050
    shocks = [FundShock(T, cap_fraction=0.08), FundShock(T, redeem_fraction=0.3)]
    calm, _ = mini_bist(seed=0)
    shocked, fund = mini_bist(seed=0, shocks=shocks)
    calm.run(9000)
    shocked.run(9000)

    def liquid_low(sim_history: list) -> float:  # type: ignore[type-arg]
        avg = [np.mean([s.assets[x].last_price for x in LIQ]) for s in sim_history]
        return min(avg[T:]) / avg[T - 1] - 1

    assert calm.history[:T] == shocked.history[:T]  # identical until the shock
    assert liquid_low(shocked.history) < liquid_low(calm.history) - 0.03
    assert calm.exchange.index_halts == []
    assert any(th == 0.05 for _, th in shocked.exchange.index_halts)
    sold = [r.sold for r in fund.records[T:]]
    liquid_sold = sum(d.get(s, 0) for d in sold for s in LIQ)
    corner_sold = sum(d.get("CORNER", 0) for d in sold)
    assert liquid_sold > corner_sold  # most of the selling lands on the liquid stocks
