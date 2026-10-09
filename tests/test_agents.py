"""Agent behaviour tests"""

from __future__ import annotations

import numpy as np
import pytest

from contagionsim.agents import MarketMaker, MarketMakerParams, NoiseTrader, NoiseTraderParams
from contagionsim.analysis import autocorr, excess_kurtosis
from contagionsim.market import Market
from contagionsim.orderbook import Side

# ---------------------------------------------------------------------------
# Market maker pricing rule (pure function, no market needed)
# ---------------------------------------------------------------------------


def test_flat_inventory_quotes_symmetric_around_reference() -> None:
    mm = MarketMaker("mm", MarketMakerParams(base_half_spread=2.0, vol_mult=0.0))
    assert mm.quotes(1000.0, inventory=0) == (998, 1002)


def test_long_inventory_skews_quotes_down() -> None:
    # Long -> wants to sell -> lower ask (easier to hit) and lower bid (buys less)
    mm = MarketMaker("mm", MarketMakerParams(base_half_spread=2.0, vol_mult=0.0, skew=0.01))
    bid0, ask0 = mm.quotes(1000.0, inventory=0)
    bid1, ask1 = mm.quotes(1000.0, inventory=500)  # reservation 995
    assert bid1 is not None and ask1 is not None and bid0 is not None and ask0 is not None
    assert bid1 < bid0 and ask1 < ask0


def test_higher_volatility_widens_spread() -> None:
    mm = MarketMaker("mm", MarketMakerParams(base_half_spread=1.0, vol_mult=1.0))
    mm.sigma2 = 1.0
    bid_lo, ask_lo = mm.quotes(1000.0, 0)
    mm.sigma2 = 25.0
    bid_hi, ask_hi = mm.quotes(1000.0, 0)
    assert ask_hi - bid_hi > ask_lo - bid_lo


@pytest.mark.parametrize(("inventory", "missing"), [(2000, "bid"), (-2000, "ask")])
def test_inventory_limit_stops_one_side(inventory: int, missing: str) -> None:
    mm = MarketMaker("mm", MarketMakerParams(max_inventory=2000))
    bid, ask = mm.quotes(1000.0, inventory)
    assert (bid is None) == (missing == "bid")
    assert (ask is None) == (missing == "ask")


# ---------------------------------------------------------------------------
# Market maker in a market
# ---------------------------------------------------------------------------


def test_market_maker_quotes_are_post_only() -> None:
    # A resting ask at 1001 would be crossed by a naive bid at 1004
    m = Market(1000)
    m.register("other")
    mm = MarketMaker("mm", MarketMakerParams(base_half_spread=1.0, vol_mult=0.0, skew=0.0))
    m.register("mm")
    m.submit_limit("other", Side.SELL, 10, 1001)
    m.last_price = 1005  # one-sided book, so the reference is this stale last price
    mm.act(0, m, np.random.default_rng(0))
    assert m.book.trades == []  # it did not take liquidity
    assert m.book.best_bid() == 1000  # bid pushed behind the 1001 ask
    m.book.check_invariants()


def test_market_maker_replaces_its_quotes_each_step() -> None:
    m = Market(1000)
    mm = MarketMaker("mm")
    m.register("mm")
    rng = np.random.default_rng(0)
    for t in range(5):
        mm.act(t, m, rng)
        assert len(m.book) == 2  # exactly one bid and one ask, old ones cancelled


# ---------------------------------------------------------------------------
# Noise trader
# ---------------------------------------------------------------------------


def test_noise_sizes_are_heavy_tailed_and_bounded() -> None:
    nt = NoiseTrader("n", NoiseTraderParams(min_size=10, size_alpha=1.6, max_size=5000))
    rng = np.random.default_rng(0)
    sizes = np.array([nt._draw_size(rng) for _ in range(20_000)])
    assert sizes.min() >= 10
    assert sizes.max() <= 5000
    assert np.median(sizes) < sizes.mean()  # right-skewed
    assert excess_kurtosis(sizes.astype(float)) > 3


def test_noise_trader_forgets_filled_and_cancelled_orders() -> None:
    m = Market(1000)
    nt = NoiseTrader("n", NoiseTraderParams(p_act=1.0, p_market=0.0, p_cancel=0.5))
    m.register("n")
    rng = np.random.default_rng(1)
    for t in range(200):
        nt.act(t, m, rng)
    # every id it still tracks is a live order it owns
    for oid in nt.live_orders:
        order = m.book.get_order(oid)
        assert order is not None and order.owner == "n"


# ---------------------------------------------------------------------------
# Stylized facts helpers
# ---------------------------------------------------------------------------


def test_excess_kurtosis_of_normal_is_near_zero() -> None:
    x = np.random.default_rng(0).standard_normal(200_000)
    assert abs(excess_kurtosis(x)) < 0.05


def test_autocorr_detects_ar1() -> None:
    rng = np.random.default_rng(0)
    x = np.zeros(50_000)
    for i in range(1, len(x)):
        x[i] = 0.5 * x[i - 1] + rng.standard_normal()
    assert autocorr(x, 1) == pytest.approx(0.5, abs=0.02)
