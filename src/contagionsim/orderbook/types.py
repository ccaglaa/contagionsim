"""Core data types for the limit order book

All prices are integer ticks (e.g. 10.05 TRY with a 0.01 tick is 1005)
Converting between ticks and currency happens at the exchange layer, never here
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Side(Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"


# eq=False: two orders are the same order only if they are the same object
# This keeps deque.remove() exact and avoids comparing every field
@dataclass(slots=True, eq=False)
class Order:
    order_id: int
    seq: int  # arrival sequence number, used for time priority
    owner: str
    side: Side
    order_type: OrderType
    quantity: int
    price: int | None  # ticks, None for market orders
    remaining: int = field(init=False)

    def __post_init__(self) -> None:
        self.remaining = self.quantity


@dataclass(frozen=True, slots=True)
class Trade:
    trade_id: int
    price: int  # ticks, always the resting order's price
    quantity: int
    buy_order_id: int
    sell_order_id: int
    buyer: str
    seller: str
    aggressor: Side | None  # side of the incoming order; None for auction trades


@dataclass(frozen=True, slots=True)
class SubmitResult:
    """What happened to one submitted order

    Conservation: quantity == filled + resting + cancelled
    """

    order_id: int
    quantity: int
    trades: tuple[Trade, ...]
    resting: int  # left in the book (limit orders only)
    cancelled: int  # unfilled and discarded (market order with no liquidity left)

    @property
    def filled(self) -> int:
        return sum(t.quantity for t in self.trades)

    @property
    def avg_price(self) -> float | None:
        if not self.trades:
            return None
        return sum(t.price * t.quantity for t in self.trades) / self.filled
