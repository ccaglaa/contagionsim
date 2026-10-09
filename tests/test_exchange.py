"""M4 exchange rules: price bounds, call auction, daily limits, circuit breakers, index"""

from __future__ import annotations

import pytest

from contagionsim.agents import MarketMaker, NoiseTrader
from contagionsim.market import (
    ANA,
    REJECTED_ID,
    YILDIZ,
    Clock,
    Exchange,
    ExchangeSimulation,
    IndexBreakerRules,
    Listing,
    Market,
    OnAsset,
)
from contagionsim.orderbook import LimitOrderBook, Side

# ---------------------------------------------------------------------------
# Book: bounds and auction mode
# ---------------------------------------------------------------------------


def bid_ladder() -> LimitOrderBook:
    book = LimitOrderBook()
    for price in (1000, 995, 990):
        book.submit_limit("b", Side.BUY, 100, price)
    return book


def test_bound_stops_a_market_order_without_auction() -> None:
    book = bid_ladder()
    res = book.submit_market("s", Side.SELL, 300, bound=995)
    assert [t.price for t in res.trades] == [1000, 995]
    assert res.cancelled == 100
    assert not book.collecting


def test_bound_with_auction_switches_to_collecting() -> None:
    book = bid_ladder()
    res = book.submit_limit("s", Side.SELL, 300, 990, bound=995, auction_on_bound=True)
    assert res.filled == 200 and res.resting == 100
    assert book.collecting
    assert book.best_bid() == 990 and book.best_ask() == 990  # crossed: allowed in an auction
    book.check_invariants()


def test_own_limit_reached_before_bound_does_not_trigger() -> None:
    book = bid_ladder()
    book.submit_limit("s", Side.SELL, 300, 998, bound=990, auction_on_bound=True)
    assert not book.collecting  # stopped by its own price, not by the rule


def test_collecting_book_does_not_match() -> None:
    book = LimitOrderBook()
    book.collecting = True
    book.submit_limit("b", Side.BUY, 100, 1010)
    res = book.submit_limit("s", Side.SELL, 100, 990)
    assert res.trades == () and res.resting == 100
    assert book.submit_market("m", Side.BUY, 50).cancelled == 50  # market orders refused


def auction_book() -> LimitOrderBook:
    """Hand example (see test below for the demand/supply table)"""
    book = LimitOrderBook()
    book.collecting = True
    for price, qty in ((1002, 100), (1000, 200), (998, 100)):
        book.submit_limit("buyer", Side.BUY, qty, price)
    for price, qty in ((997, 150), (999, 100), (1001, 200)):
        book.submit_limit("seller", Side.SELL, qty, price)
    return book


def test_auction_price_maximizes_volume() -> None:
    #   p     demand(>=p)  supply(<=p)  volume
    #  997       400          150        150
    #  998       400          150        150
    #  999       300          250        250   <- max volume, tie with 1000
    # 1000       300          250        250
    # 1001       100          450        100
    # 1002       100          450        100
    # Same volume and imbalance at 999 and 1000: closest to the reference decides
    book = auction_book()
    assert book.auction_price(reference=1000) == (1000, 250)
    assert book.auction_price(reference=990) == (999, 250)


def test_uncross_executes_at_one_price_and_resumes() -> None:
    book = auction_book()
    trades = book.uncross(reference=990)
    assert {t.price for t in trades} == {999}
    assert sum(t.quantity for t in trades) == 250
    assert all(t.aggressor is None for t in trades)
    assert not book.collecting
    assert (book.best_bid(), book.best_ask()) == (1000, 1001)
    assert book.depth(Side.BUY) == [(1000, 50), (998, 100)]  # price-time priority
    book.check_invariants()


def test_uncross_without_cross_just_resumes() -> None:
    book = LimitOrderBook()
    book.collecting = True
    book.submit_limit("b", Side.BUY, 10, 990)
    book.submit_limit("s", Side.SELL, 10, 1010)
    assert book.uncross() == []
    assert not book.collecting


# ---------------------------------------------------------------------------
# Market with Ana Pazar rules: limit 15%, circuit breaker 7.5%, 15+2 min auction
# ---------------------------------------------------------------------------


def ana_market() -> Market:
    m = Market(1000, rules=ANA, clock=Clock(steps_per_day=1000, steps_per_minute=2))
    for name in ("b", "s"):
        m.register(name)
    return m


def test_daily_band_from_base_price() -> None:
    m = ana_market()
    assert (m.floor, m.ceiling) == (850, 1150)


def test_order_outside_band_is_rejected() -> None:
    m = ana_market()
    res = m.submit_limit("s", Side.SELL, 10, 849)
    assert res.order_id == REJECTED_ID and res.cancelled == 10
    assert len(m.book) == 0


def test_circuit_breaker_halts_the_stock() -> None:
    m = ana_market()
    for price in (990, 950, 920):  # 920 is beyond 7.5% from 1000 (bound 925)
        m.submit_limit("b", Side.BUY, 100, price)
    res = m.submit_market("s", Side.SELL, 300)
    assert [t.price for t in res.trades] == [990, 950]
    assert m.halted and m.halts == [0]
    assert m.auction_end == 34  # (15 + 2) minutes x 2 steps per minute


def test_auction_ends_on_time_and_resets_reference() -> None:
    m = ana_market()
    m.submit_limit("b", Side.BUY, 100, 920)
    m.submit_limit("s", Side.SELL, 50, 900)  # hits the 925 bound -> halt, rests in auction
    assert m.halted
    m.t = 33
    assert m.maybe_end_auction() == [] and m.halted
    m.t = 34
    trades = m.maybe_end_auction()
    assert [(t.price, t.quantity) for t in trades] == [(920, 50)]
    assert not m.halted
    assert m.cb_reference == 920  # next circuit breaker measured from the auction price
    assert m.accounts["s"].position == -50 and m.accounts["b"].position == 50


def test_locked_at_the_floor_when_limit_is_tighter_than_breaker() -> None:
    m = ana_market()
    m.cb_reference = 900  # breaker band would be 833, below the 850 floor
    m.submit_limit("b", Side.BUY, 10, 860)
    res = m.submit_market("s", Side.SELL, 100)
    assert [t.price for t in res.trades] == [860]
    assert res.cancelled == 90
    assert not m.halted  # no auction: the stock is simply limit-down


def test_new_day_moves_the_band_and_drops_stale_orders() -> None:
    m = ana_market()
    m.submit_limit("b", Side.BUY, 10, 860)
    m.last_price = 900
    m.start_day()
    assert (m.base_price, m.floor, m.ceiling) == (900, 765, 1035)
    assert len(m.book) == 1  # 860 is still inside the new band
    m.last_price = 1100
    m.start_day()  # band 935..1265: the 860 bid is outside and cancelled
    assert len(m.book) == 0


def test_market_without_rules_behaves_like_m1_to_m3() -> None:
    m = Market(1000)
    m.register("b")
    m.register("s")
    m.submit_limit("b", Side.BUY, 100, 10)
    assert m.submit_market("s", Side.SELL, 100).filled == 100  # no limits at all


# ---------------------------------------------------------------------------
# Exchange: shared cash, index, index circuit breaker, day roll
# ---------------------------------------------------------------------------


def two_stock_exchange() -> Exchange:
    return Exchange(
        [
            Listing("BIG", 1000, shares_outstanding=900, rules=YILDIZ),
            Listing("SMALL", 1000, shares_outstanding=100, rules=ANA),
        ],
        index_rules=IndexBreakerRules(),
        clock=Clock(steps_per_day=1000, steps_per_minute=2),
    )


def test_cash_is_shared_across_markets() -> None:
    ex = two_stock_exchange()
    ex.register("x", cash=1_000_000)
    ex.register("y", cash=1_000_000)
    ex.begin_step(0)
    ex["BIG"].submit_limit("y", Side.SELL, 10, 1000)
    ex["BIG"].submit_market("x", Side.BUY, 10)
    ex["SMALL"].submit_limit("y", Side.BUY, 5, 1000)
    ex["SMALL"].submit_market("x", Side.SELL, 5)
    assert ex.cash("x") == 1_000_000 - 10 * 1000 + 5 * 1000
    assert ex.position("x", "BIG") == 10 and ex.position("x", "SMALL") == -5
    assert ex.cash("x") + ex.cash("y") == ex.ledger.external_cash


def test_index_is_cap_weighted() -> None:
    ex = two_stock_exchange()
    assert ex.index_level() == 100.0
    ex["SMALL"].last_price = 500  # small stock halves: weight 10% -> index -5%
    assert ex.index_level() == pytest.approx(95.0)


def test_index_breaker_halts_everything_once_per_stage() -> None:
    ex = two_stock_exchange()
    ex.begin_step(0)
    ex["BIG"].last_price = 940  # index 94.6: -5.4%
    ex.end_step(0)
    assert ex.index_halts == [(0, 0.05)]
    assert all(m.halted for m in ex.markets.values())
    assert ex["BIG"].auction_end == 60  # (20 + 5 + 5) minutes x 2
    ex.begin_step(60)  # auctions end (nothing to cross)
    assert not any(m.halted for m in ex.markets.values())
    ex.end_step(60)
    assert len(ex.index_halts) == 1  # the 5% stage does not fire twice
    ex["BIG"].last_price = 920  # -7.2%: second stage
    ex.end_step(61)
    assert ex.index_halts[-1] == (61, 0.07)


def test_new_day_resets_index_reference_and_stages() -> None:
    ex = two_stock_exchange()
    ex.begin_step(0)
    ex["BIG"].last_price = 940
    ex.end_step(0)
    ex.begin_step(1000)
    assert ex.index_prev_close == pytest.approx(94.6)
    assert ex.index_stages_fired == set()
    assert ex["BIG"].base_price == 940


# ---------------------------------------------------------------------------
# Full simulation on two stocks
# ---------------------------------------------------------------------------


def exchange_sim(seed: int) -> ExchangeSimulation:
    ex = two_stock_exchange()
    agents = []
    for sym in ("BIG", "SMALL"):
        agents += [OnAsset(sym, NoiseTrader(f"noise_{sym}_{i}")) for i in range(10)]
        agents.append(OnAsset(sym, MarketMaker(f"mm_{sym}")))
    return ExchangeSimulation(ex, agents, seed=seed)  # type: ignore[arg-type]


def test_exchange_simulation_is_valid_and_reproducible() -> None:
    sim = exchange_sim(seed=5)
    for _ in range(2500):  # crosses two day boundaries
        snap = sim.step()
        for m in sim.exchange.markets.values():
            m.book.check_invariants()
            assert m.floor <= m.last_price <= m.ceiling
        assert snap.index > 0
    ledger = sim.exchange.ledger
    assert sum(ledger.cash.values()) == ledger.external_cash
    for sym in ("BIG", "SMALL"):
        assert sum(p.get(sym, 0) for p in ledger.positions.values()) == 0
    again = exchange_sim(seed=5).run(300)
    assert again == sim.history[:300]
