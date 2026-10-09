"""Limit order book with price-time priority matching

Structure per side:
    _levels[side]: dict price -> deque of resting orders (FIFO = time priority)
    _prices[side]: sorted list of prices that have at least one order
Plus _orders: dict order_id -> Order, so cancel is a dict lookup, not a scan

Invariants after every public call (see check_invariants):
    best bid < best ask whenever both exist (book never crossed)
    no empty price levels
    every resting order has remaining > 0 and is in _orders
"""

from __future__ import annotations

from bisect import bisect_left, insort
from collections import deque

from contagionsim.orderbook.types import Order, OrderType, Side, SubmitResult, Trade


class LimitOrderBook:
    def __init__(self, symbol: str = "ASSET") -> None:
        self.symbol = symbol
        self._levels: dict[Side, dict[int, deque[Order]]] = {Side.BUY: {}, Side.SELL: {}}
        self._prices: dict[Side, list[int]] = {Side.BUY: [], Side.SELL: []}
        self._orders: dict[int, Order] = {}
        self.trades: list[Trade] = []
        # Integer counters instead of wall-clock time: deterministic and reproducible
        self._next_order_id = 1
        self._next_seq = 1
        self._next_trade_id = 1

    # ------------------------------------------------------------------
    # Public API: order entry
    # ------------------------------------------------------------------

    def submit_limit(self, owner: str, side: Side, quantity: int, price: int) -> SubmitResult:
        """Match as much as possible, then rest the remainder at `price`"""
        _check_quantity(quantity)
        if not isinstance(price, int) or isinstance(price, bool) or price <= 0:
            raise ValueError(f"price must be a positive int (ticks), got {price!r}")
        order = self._new_order(owner, side, OrderType.LIMIT, quantity, price)
        trades = self._match(order)
        resting = order.remaining
        if resting > 0:
            self._add_resting(order)
        return SubmitResult(order.order_id, quantity, tuple(trades), resting, 0)

    def submit_market(self, owner: str, side: Side, quantity: int) -> SubmitResult:
        """Match against the book; any unfilled remainder is cancelled, never rests"""
        _check_quantity(quantity)
        order = self._new_order(owner, side, OrderType.MARKET, quantity, None)
        trades = self._match(order)
        return SubmitResult(order.order_id, quantity, tuple(trades), 0, order.remaining)

    def cancel(self, order_id: int) -> bool:
        """Remove a resting order. Returns False if it is not in the book

        Not an error: in a simulation an agent often cancels an order that
        was filled a moment earlier, so this must be safe to call
        """
        order = self._orders.pop(order_id, None)
        if order is None:
            return False
        assert order.price is not None
        queue = self._levels[order.side][order.price]
        queue.remove(order)
        if not queue:
            self._remove_level(order.side, order.price)
        return True

    # ------------------------------------------------------------------
    # Public API: market data
    # ------------------------------------------------------------------

    def best_bid(self) -> int | None:
        prices = self._prices[Side.BUY]
        return prices[-1] if prices else None

    def best_ask(self) -> int | None:
        prices = self._prices[Side.SELL]
        return prices[0] if prices else None

    def spread(self) -> int | None:
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return ask - bid

    def mid(self) -> float | None:
        # float, because the mid can fall on a half tick
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2

    def depth(self, side: Side, n_levels: int = 5) -> list[tuple[int, int]]:
        """Top n price levels as (price, total quantity), best first"""
        prices = self._prices[side]
        best_first = reversed(prices) if side is Side.BUY else iter(prices)
        out: list[tuple[int, int]] = []
        for price in best_first:
            if len(out) == n_levels:
                break
            out.append((price, sum(o.remaining for o in self._levels[side][price])))
        return out

    def imbalance(self, n_levels: int = 1) -> float | None:
        """(bid vol - ask vol) / (bid vol + ask vol) over the top n levels, in [-1, 1]

        None if both sides are empty. One-sided book gives +1 or -1
        """
        bid_vol = sum(q for _, q in self.depth(Side.BUY, n_levels))
        ask_vol = sum(q for _, q in self.depth(Side.SELL, n_levels))
        total = bid_vol + ask_vol
        if total == 0:
            return None
        return (bid_vol - ask_vol) / total

    def get_order(self, order_id: int) -> Order | None:
        return self._orders.get(order_id)

    def __len__(self) -> int:
        """Number of resting orders"""
        return len(self._orders)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _new_order(
        self, owner: str, side: Side, order_type: OrderType, quantity: int, price: int | None
    ) -> Order:
        order = Order(self._next_order_id, self._next_seq, owner, side, order_type, quantity, price)
        self._next_order_id += 1
        self._next_seq += 1
        return order

    def _match(self, order: Order) -> list[Trade]:
        """Walk the opposite side best price first, FIFO within each level"""
        opp = order.side.opposite
        trades: list[Trade] = []
        while order.remaining > 0:
            best = self.best_ask() if opp is Side.SELL else self.best_bid()
            if best is None:
                break  # opposite side is empty
            if order.order_type is OrderType.LIMIT and not _crosses(order, best):
                break  # best opposite price is worse than our limit
            queue = self._levels[opp][best]
            resting = queue[0]
            qty = min(order.remaining, resting.remaining)
            trades.append(self._record_trade(order, resting, best, qty))
            order.remaining -= qty
            resting.remaining -= qty
            if resting.remaining == 0:
                queue.popleft()
                del self._orders[resting.order_id]
                if not queue:
                    self._remove_level(opp, best)
        return trades

    def _record_trade(self, aggressor: Order, resting: Order, price: int, qty: int) -> Trade:
        buy, sell = (aggressor, resting) if aggressor.side is Side.BUY else (resting, aggressor)
        trade = Trade(
            trade_id=self._next_trade_id,
            price=price,
            quantity=qty,
            buy_order_id=buy.order_id,
            sell_order_id=sell.order_id,
            buyer=buy.owner,
            seller=sell.owner,
            aggressor=aggressor.side,
        )
        self._next_trade_id += 1
        self.trades.append(trade)
        return trade

    def _add_resting(self, order: Order) -> None:
        assert order.price is not None
        levels = self._levels[order.side]
        if order.price not in levels:
            levels[order.price] = deque()
            insort(self._prices[order.side], order.price)
        levels[order.price].append(order)
        self._orders[order.order_id] = order

    def _remove_level(self, side: Side, price: int) -> None:
        del self._levels[side][price]
        prices = self._prices[side]
        prices.pop(bisect_left(prices, price))

    def check_invariants(self) -> None:
        """Raise AssertionError if the book is in an impossible state (used by tests)"""
        bid, ask = self.best_bid(), self.best_ask()
        if bid is not None and ask is not None:
            assert bid < ask, f"crossed book: bid {bid} >= ask {ask}"
        seen = 0
        for side in Side:
            assert self._prices[side] == sorted(self._levels[side]), "price index out of sync"
            for price, queue in self._levels[side].items():
                assert queue, f"empty level {side} {price}"
                last_seq = 0
                for o in queue:
                    assert o.remaining > 0
                    assert o.price == price and o.side is side
                    assert self._orders.get(o.order_id) is o
                    assert o.seq > last_seq, "FIFO order broken"
                    last_seq = o.seq
                    seen += 1
        assert seen == len(self._orders), "order index has stale entries"


def _crosses(order: Order, best_opposite: int) -> bool:
    assert order.price is not None
    if order.side is Side.BUY:
        return order.price >= best_opposite
    return order.price <= best_opposite


def _check_quantity(quantity: int) -> None:
    # Raise on bad input: a zero or negative order is always a bug in the caller
    if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
        raise ValueError(f"quantity must be a positive int, got {quantity!r}")
