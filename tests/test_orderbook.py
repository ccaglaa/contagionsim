"""Order book tests

Prices are ticks with a 0.01 tick: 1005 means 10.05
The `lesson_book` fixture is the book from M1 lesson 1:

    asks  1010 x 500 (C, t2)
          1005 x 300 (A, t1), 1005 x 200 (B, t3)
    bids  1000 x 400 (D, t4)
           995 x 100 (E, t5)
"""

from __future__ import annotations

import random

import pytest

from contagionsim.orderbook import LimitOrderBook, Side


@pytest.fixture
def lesson_book() -> tuple[LimitOrderBook, dict[str, int]]:
    book = LimitOrderBook()
    ids = {
        "A": book.submit_limit("A", Side.SELL, 300, 1005).order_id,
        "C": book.submit_limit("C", Side.SELL, 500, 1010).order_id,
        "B": book.submit_limit("B", Side.SELL, 200, 1005).order_id,
        "D": book.submit_limit("D", Side.BUY, 400, 1000).order_id,
        "E": book.submit_limit("E", Side.BUY, 100, 995).order_id,
    }
    book.check_invariants()
    return book, ids


# ---------------------------------------------------------------------------
# Empty book and market data
# ---------------------------------------------------------------------------


def test_empty_book_has_no_quotes() -> None:
    book = LimitOrderBook()
    assert book.best_bid() is None
    assert book.best_ask() is None
    assert book.spread() is None
    assert book.mid() is None
    assert book.imbalance() is None
    assert len(book) == 0


def test_lesson_book_top_of_book(lesson_book) -> None:
    book, _ = lesson_book
    assert book.best_bid() == 1000
    assert book.best_ask() == 1005
    assert book.spread() == 5
    assert book.mid() == 1002.5
    assert book.depth(Side.SELL) == [(1005, 500), (1010, 500)]
    assert book.depth(Side.BUY) == [(1000, 400), (995, 100)]
    # top level only: (400 - 500) / 900
    assert book.imbalance(1) == pytest.approx(-100 / 900)


def test_one_sided_book_imbalance_is_plus_or_minus_one() -> None:
    book = LimitOrderBook()
    book.submit_limit("x", Side.BUY, 10, 100)
    assert book.imbalance() == 1.0
    assert book.spread() is None


# ---------------------------------------------------------------------------
# Matching: the worked examples from the lesson
# ---------------------------------------------------------------------------


def test_market_buy_walks_the_book(lesson_book) -> None:
    book, ids = lesson_book
    res = book.submit_market("buyer", Side.BUY, 600)

    got = [(t.price, t.quantity, t.sell_order_id) for t in res.trades]
    assert got == [(1005, 300, ids["A"]), (1005, 200, ids["B"]), (1010, 100, ids["C"])]
    assert res.filled == 600 and res.cancelled == 0 and res.resting == 0
    assert book.get_order(ids["C"]).remaining == 400  # partial fill
    assert book.best_ask() == 1010
    assert book.spread() == 10
    book.check_invariants()


def test_aggressive_limit_sell_fills_then_rests(lesson_book) -> None:
    # Exercise 1: limit sell 500 @ 9.98
    book, ids = lesson_book
    res = book.submit_limit("seller", Side.SELL, 500, 998)

    assert [(t.price, t.quantity, t.buy_order_id) for t in res.trades] == [(1000, 400, ids["D"])]
    assert res.resting == 100
    assert book.best_bid() == 995
    assert book.best_ask() == 998
    assert book.spread() == 3
    assert book.imbalance(1) == 0.0
    book.check_invariants()


def test_limit_sell_can_empty_the_bid_side(lesson_book) -> None:
    # Quick check: limit sell 500 @ 9.95 takes D then E, nothing rests
    book, ids = lesson_book
    res = book.submit_limit("seller", Side.SELL, 500, 995)

    assert [(t.price, t.quantity) for t in res.trades] == [(1000, 400), (995, 100)]
    assert res.resting == 0
    assert book.best_bid() is None
    assert book.spread() is None
    assert book.imbalance() == -1.0
    book.check_invariants()


def test_trade_price_is_resting_price_not_incoming(lesson_book) -> None:
    # Buyer is willing to pay 10.10 but the resting ask is 10.05
    book, _ = lesson_book
    res = book.submit_limit("buyer", Side.BUY, 100, 1010)
    assert res.trades[0].price == 1005


def test_time_priority_within_level() -> None:
    book = LimitOrderBook()
    first = book.submit_limit("first", Side.BUY, 100, 500).order_id
    second = book.submit_limit("second", Side.BUY, 100, 500).order_id
    res = book.submit_market("s", Side.SELL, 150)
    assert [(t.buy_order_id, t.quantity) for t in res.trades] == [(first, 100), (second, 50)]


def test_price_priority_beats_time_priority() -> None:
    book = LimitOrderBook()
    book.submit_limit("early", Side.SELL, 100, 1010)
    late_better = book.submit_limit("late", Side.SELL, 100, 1005).order_id
    res = book.submit_market("b", Side.BUY, 50)
    assert res.trades[0].sell_order_id == late_better


def test_non_crossing_limit_rests_without_trading(lesson_book) -> None:
    book, _ = lesson_book
    res = book.submit_limit("x", Side.BUY, 50, 1001)  # inside the spread
    assert res.trades == () and res.resting == 50
    assert book.best_bid() == 1001
    book.check_invariants()


def test_trade_records_owners_and_aggressor(lesson_book) -> None:
    book, _ = lesson_book
    t = book.submit_market("fund", Side.SELL, 10).trades[0]
    assert (t.buyer, t.seller, t.aggressor) == ("D", "fund", Side.SELL)
    assert book.trades[-1] is t


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_market_order_on_empty_side_is_cancelled() -> None:
    book = LimitOrderBook()
    res = book.submit_market("x", Side.SELL, 100)
    assert res.trades == () and res.cancelled == 100
    assert len(book) == 0  # market orders never rest


def test_market_order_larger_than_book_partially_fills(lesson_book) -> None:
    book, _ = lesson_book
    res = book.submit_market("fund", Side.SELL, 1000)  # only 500 on the bid side
    assert res.filled == 500 and res.cancelled == 500
    assert book.best_bid() is None
    assert res.avg_price == pytest.approx((1000 * 400 + 995 * 100) / 500)


@pytest.mark.parametrize("qty", [0, -5, 1.5, True])
def test_bad_quantity_raises(qty) -> None:
    book = LimitOrderBook()
    with pytest.raises(ValueError):
        book.submit_limit("x", Side.BUY, qty, 100)
    with pytest.raises(ValueError):
        book.submit_market("x", Side.BUY, qty)


@pytest.mark.parametrize("price", [0, -1, 10.05])
def test_bad_price_raises(price) -> None:
    with pytest.raises(ValueError):
        LimitOrderBook().submit_limit("x", Side.BUY, 10, price)


def test_bad_input_does_not_consume_an_id() -> None:
    book = LimitOrderBook()
    with pytest.raises(ValueError):
        book.submit_limit("x", Side.BUY, 0, 100)
    assert book.submit_limit("x", Side.BUY, 1, 100).order_id == 1


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


def test_cancel_removes_order_and_empty_level(lesson_book) -> None:
    book, ids = lesson_book
    assert book.cancel(ids["D"]) is True
    assert book.best_bid() == 995
    assert book.get_order(ids["D"]) is None
    book.check_invariants()


def test_cancel_keeps_other_orders_at_same_level(lesson_book) -> None:
    book, ids = lesson_book
    book.cancel(ids["A"])
    assert book.depth(Side.SELL, 1) == [(1005, 200)]
    res = book.submit_market("b", Side.BUY, 10)
    assert res.trades[0].sell_order_id == ids["B"]


def test_cancel_unknown_or_filled_order_returns_false(lesson_book) -> None:
    book, ids = lesson_book
    assert book.cancel(999) is False
    book.submit_market("s", Side.SELL, 400)  # fills D completely
    assert book.cancel(ids["D"]) is False
    assert book.cancel(ids["E"]) is True
    assert book.cancel(ids["E"]) is False  # double cancel


def test_cancel_partially_filled_order(lesson_book) -> None:
    book, ids = lesson_book
    book.submit_market("b", Side.BUY, 600)  # leaves C with 400
    assert book.cancel(ids["C"]) is True
    assert book.best_ask() is None
    book.check_invariants()


# ---------------------------------------------------------------------------
# Randomized stress test: thousands of random operations, invariants every step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_random_operations_preserve_invariants(seed: int) -> None:
    rng = random.Random(seed)
    book = LimitOrderBook()
    limit_qty: dict[int, int] = {}  # order_id -> submitted quantity
    cancelled: dict[int, int] = {}  # order_id -> quantity removed by cancel

    for _ in range(3000):
        r = rng.random()
        side = rng.choice([Side.BUY, Side.SELL])
        qty = rng.randint(1, 200)
        if r < 0.6:
            res = book.submit_limit("L", side, qty, rng.randint(990, 1010))
            limit_qty[res.order_id] = qty
            assert res.filled + res.resting == qty
        elif r < 0.8:
            res = book.submit_market("M", side, qty)
            assert res.filled + res.cancelled == qty
        elif limit_qty:
            oid = rng.choice(list(limit_qty))
            order = book.get_order(oid)
            was_resting = order.remaining if order is not None else 0
            assert book.cancel(oid) is (order is not None)
            cancelled[oid] = cancelled.get(oid, 0) + was_resting
        book.check_invariants()

    # Conservation per limit order: submitted = filled + cancelled + still resting
    filled: dict[int, int] = {}
    for t in book.trades:
        for oid in (t.buy_order_id, t.sell_order_id):
            filled[oid] = filled.get(oid, 0) + t.quantity
    for oid, qty in limit_qty.items():
        order = book.get_order(oid)
        still = order.remaining if order is not None else 0
        assert qty == filled.get(oid, 0) + cancelled.get(oid, 0) + still
