"""Single-asset market: an order book plus account bookkeeping for every agent

Agents never touch the book directly. They go through Market, which:
    - forwards orders to the book
    - settles every trade into the buyer and seller accounts (cash and position)
    - remembers the last trade price (used when the book is one-sided or empty)

Units: cash is in ticks x shares (integer), so no rounding ever happens
"""

from __future__ import annotations

from dataclasses import dataclass

from contagionsim.orderbook import LimitOrderBook, Side, SubmitResult, Trade


@dataclass(slots=True)
class Account:
    cash: int = 0  # ticks x shares
    position: int = 0  # shares, negative = short

    def value(self, price: float) -> float:
        """Mark-to-market value at `price` (ticks)"""
        return self.cash + self.position * price


class Market:
    def __init__(self, initial_price: int, symbol: str = "ASSET") -> None:
        if initial_price <= 0:
            raise ValueError("initial_price must be positive")
        self.book = LimitOrderBook(symbol)
        self.accounts: dict[str, Account] = {}
        self.last_price: int = initial_price

    def register(self, name: str, cash: int = 0, position: int = 0) -> Account:
        if name in self.accounts:
            raise ValueError(f"agent {name!r} already registered")
        account = Account(cash, position)
        self.accounts[name] = account
        return account

    # ------------------------------------------------------------------
    # Order entry (same signatures as the book, plus settlement)
    # ------------------------------------------------------------------

    def submit_limit(self, owner: str, side: Side, quantity: int, price: int) -> SubmitResult:
        self._check_registered(owner)
        result = self.book.submit_limit(owner, side, quantity, price)
        self._settle(result.trades)
        return result

    def submit_market(self, owner: str, side: Side, quantity: int) -> SubmitResult:
        self._check_registered(owner)
        result = self.book.submit_market(owner, side, quantity)
        self._settle(result.trades)
        return result

    def cancel(self, owner: str, order_id: int) -> bool:
        """Cancel one of `owner`'s orders. False if already filled or cancelled"""
        order = self.book.get_order(order_id)
        if order is not None and order.owner != owner:
            # Cancelling someone else's order is always a bug in the caller
            raise PermissionError(f"{owner!r} cannot cancel order {order_id} of {order.owner!r}")
        return self.book.cancel(order_id)

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    def reference_price(self) -> float:
        """Mid if both sides are quoted, otherwise the last trade price

        Agents use this as "the price". It always exists, even on an empty book
        """
        mid = self.book.mid()
        return mid if mid is not None else float(self.last_price)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _settle(self, trades: tuple[Trade, ...]) -> None:
        for t in trades:
            notional = t.price * t.quantity
            buyer = self.accounts[t.buyer]
            seller = self.accounts[t.seller]
            buyer.position += t.quantity
            buyer.cash -= notional
            seller.position -= t.quantity
            seller.cash += notional
            self.last_price = t.price

    def _check_registered(self, owner: str) -> None:
        if owner not in self.accounts:
            raise KeyError(f"agent {owner!r} is not registered")
