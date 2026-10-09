"""One tradable asset: an order book, account bookkeeping and (optionally) exchange rules

Agents never touch the book directly. They go through Market, which:
    - forwards orders to the book
    - settles every trade into the buyer and seller accounts (cash and position)
    - remembers the last trade price (used when the book is one-sided or empty)
    - if `rules` are given (M4): enforces the daily price limit and the
      stock circuit breaker, and runs the call auction that follows a halt

Accounts live in a Ledger. A standalone Market has its own Ledger (M1-M3).
In a multi-asset Exchange all markets share one Ledger, so an agent has ONE
cash balance and a position per asset

Units: cash is in ticks x shares (integer), so no rounding ever happens
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping

from contagionsim.market.rules import Clock, SegmentRules
from contagionsim.orderbook import LimitOrderBook, Side, SubmitResult, Trade

REJECTED_ID = -1  # order_id of an order refused by exchange rules (never entered the book)

# ---------------------------------------------------------------------------
# Ledger and account views
# ---------------------------------------------------------------------------


class Ledger:
    """Cash per agent and positions per (agent, symbol)

    external_cash: cash that entered or left from outside the market (initial
    cash, fund subscriptions and redemptions). Conservation:
        sum of all cash == external_cash
    """

    def __init__(self) -> None:
        self.cash: dict[str, int] = {}
        self.positions: dict[str, dict[str, int]] = {}
        self.external_cash = 0

    def register(self, name: str, cash: int = 0) -> None:
        if name in self.cash:
            raise ValueError(f"agent {name!r} already registered")
        self.cash[name] = cash
        self.positions[name] = {}
        self.external_cash += cash

    def transfer_external(self, name: str, amount: int) -> None:
        if name not in self.cash:
            raise KeyError(f"agent {name!r} is not registered")
        if self.cash[name] + amount < 0:
            raise ValueError(f"{name!r} has {self.cash[name]} cash, cannot pay out {-amount}")
        self.cash[name] += amount
        self.external_cash += amount


class Account:
    """An agent's account as seen from one market: shared cash, this symbol's position"""

    __slots__ = ("_ledger", "name", "symbol")

    def __init__(self, ledger: Ledger, name: str, symbol: str) -> None:
        self._ledger = ledger
        self.name = name
        self.symbol = symbol

    @property
    def cash(self) -> int:
        return self._ledger.cash[self.name]

    @cash.setter
    def cash(self, value: int) -> None:
        self._ledger.cash[self.name] = value

    @property
    def position(self) -> int:
        return self._ledger.positions[self.name].get(self.symbol, 0)

    @position.setter
    def position(self, value: int) -> None:
        self._ledger.positions[self.name][self.symbol] = value

    def value(self, price: float) -> float:
        """Cash + this position at `price` (ticks). Ignores positions in other symbols"""
        return self.cash + self.position * price


class Accounts(Mapping[str, Account]):
    """Read-only mapping name -> Account for one symbol"""

    def __init__(self, ledger: Ledger, symbol: str) -> None:
        self._ledger = ledger
        self._symbol = symbol

    def __getitem__(self, name: str) -> Account:
        if name not in self._ledger.cash:
            raise KeyError(name)
        return Account(self._ledger, name, self._symbol)

    def __iter__(self) -> Iterator[str]:
        return iter(self._ledger.cash)

    def __len__(self) -> int:
        return len(self._ledger.cash)


# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------


class Market:
    def __init__(
        self,
        initial_price: int,
        symbol: str = "ASSET",
        rules: SegmentRules | None = None,
        clock: Clock | None = None,
        ledger: Ledger | None = None,
    ) -> None:
        if initial_price <= 0:
            raise ValueError("initial_price must be positive")
        self.symbol = symbol
        self.book = LimitOrderBook(symbol)
        self.ledger = ledger or Ledger()
        self.accounts = Accounts(self.ledger, symbol)
        self.last_price: int = initial_price
        self.rules = rules
        self.clock = clock or Clock()
        self.t = 0  # set by the Exchange each step
        # Rule state, set by start_day()
        self.base_price = initial_price  # previous close
        self.floor = 1
        self.ceiling = 2**62
        self.cb_reference = initial_price  # circuit breaker moves are measured from here
        self.auction_end: int | None = None  # step at which the running auction uncrosses
        self.halts: list[int] = []  # steps at which a stock circuit breaker fired
        if rules is not None:
            self.start_day()

    @property
    def external_cash(self) -> int:
        return self.ledger.external_cash

    def register(self, name: str, cash: int = 0, position: int = 0) -> Account:
        self.ledger.register(name, cash)
        account = self.accounts[name]
        if position:
            account.position = position
        return account

    def transfer_external(self, name: str, amount: int) -> None:
        """Move cash between an account and the outside world (+ in, - out)

        Used for fund subscriptions (+) and redemption payouts (-)
        An account can never be pushed below zero cash by a transfer
        """
        self.ledger.transfer_external(name, amount)

    # ------------------------------------------------------------------
    # Order entry
    # ------------------------------------------------------------------

    def submit_limit(self, owner: str, side: Side, quantity: int, price: int) -> SubmitResult:
        self._check_registered(owner)
        if self.rules is not None and not self.floor <= price <= self.ceiling:
            # Outside the daily price limit: refused by the exchange (a normal event)
            return SubmitResult(REJECTED_ID, quantity, (), 0, quantity)
        bound, auction = self._bound(side)
        was_collecting = self.book.collecting
        result = self.book.submit_limit(owner, side, quantity, price, bound, auction)
        self._after_order(result, was_collecting)
        return result

    def submit_market(self, owner: str, side: Side, quantity: int) -> SubmitResult:
        self._check_registered(owner)
        bound, auction = self._bound(side)
        was_collecting = self.book.collecting
        result = self.book.submit_market(owner, side, quantity, bound, auction)
        self._after_order(result, was_collecting)
        return result

    def cancel(self, owner: str, order_id: int) -> bool:
        """Cancel one of `owner`'s orders. False if already filled, cancelled or rejected"""
        order = self.book.get_order(order_id)
        if order is not None and order.owner != owner:
            # Cancelling someone else's order is always a bug in the caller
            raise PermissionError(f"{owner!r} cannot cancel order {order_id} of {order.owner!r}")
        return self.book.cancel(order_id)

    # ------------------------------------------------------------------
    # Prices and state
    # ------------------------------------------------------------------

    def reference_price(self) -> float:
        """Mid if both sides are quoted (and not in an auction), else the last trade price

        Agents use this as "the price". It always exists, even on an empty book
        """
        mid = self.book.mid()
        if mid is None or self.book.collecting:
            return float(self.last_price)
        return mid

    @property
    def halted(self) -> bool:
        return self.book.collecting

    def at_floor(self) -> bool:
        """True if the last trade happened at the daily lower limit (limit-down)"""
        return self.rules is not None and self.last_price <= self.floor

    # ------------------------------------------------------------------
    # Exchange rules (called by the Exchange)
    # ------------------------------------------------------------------

    def start_day(self) -> None:
        """New session: base price = last close, new price limits, circuit breaker reset"""
        if self.book.collecting:
            self.end_auction()
        self.base_price = self.last_price
        self.cb_reference = self.last_price
        if self.rules is not None:
            limit = self.rules.daily_limit
            self.floor = max(1, math.ceil(self.base_price * (1 - limit)))
            self.ceiling = math.floor(self.base_price * (1 + limit))
            self.book.cancel_outside(self.floor, self.ceiling)

    def start_auction(self, steps: int) -> None:
        """Halt continuous trading for `steps` steps (orders collect, nothing trades)"""
        self.book.collecting = True
        end = self.t + steps
        self.auction_end = end if self.auction_end is None else max(self.auction_end, end)

    def maybe_end_auction(self) -> list[Trade]:
        if self.auction_end is not None and self.t >= self.auction_end:
            return self.end_auction()
        return []

    def end_auction(self) -> list[Trade]:
        trades = self.book.uncross(reference=float(self.cb_reference))
        self._settle(tuple(trades))
        if trades:
            self.cb_reference = trades[0].price  # the auction price is the new reference
        self.auction_end = None
        return trades

    def _bound(self, side: Side) -> tuple[int | None, bool]:
        """Worst price an incoming order may trade at, and whether hitting it starts an auction

        The circuit-breaker band (cb_reference +/- trigger) is tighter than the daily
        limit: hitting it starts an auction. If the daily limit is tighter (late in a
        big move), hitting it just stops the order: the stock is locked at the limit
        """
        if self.rules is None:
            return None, False
        trig = self.rules.cb_trigger
        if side is Side.SELL:
            cb = math.ceil(self.cb_reference * (1 - trig))
            return (cb, True) if cb > self.floor else (self.floor, False)
        cb = math.floor(self.cb_reference * (1 + trig))
        return (cb, True) if cb < self.ceiling else (self.ceiling, False)

    def _after_order(self, result: SubmitResult, was_collecting: bool) -> None:
        self._settle(result.trades)
        if self.rules is not None and self.book.collecting and not was_collecting:
            # This order hit the circuit-breaker band: volatility interruption
            self.halts.append(self.t)
            self.auction_end = None
            self.start_auction(self.clock.minutes_to_steps(self.rules.auction_minutes()))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _settle(self, trades: tuple[Trade, ...]) -> None:
        cash, pos = self.ledger.cash, self.ledger.positions
        for t in trades:
            notional = t.price * t.quantity
            cash[t.buyer] -= notional
            cash[t.seller] += notional
            pos[t.buyer][self.symbol] = pos[t.buyer].get(self.symbol, 0) + t.quantity
            pos[t.seller][self.symbol] = pos[t.seller].get(self.symbol, 0) - t.quantity
            self.last_price = t.price

    def _check_registered(self, owner: str) -> None:
        if owner not in self.ledger.cash:
            raise KeyError(f"agent {owner!r} is not registered")
