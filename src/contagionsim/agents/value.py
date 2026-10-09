"""Value trader: trades toward a fundamental value, with limited capacity

Each step, with probability p_act, it compares the reference price with its
fundamental value V:
    mispricing = (reference - V) / V
    |mispricing| < threshold  -> do nothing (not worth the risk)
    price too high            -> sell
    price too low             -> buy
Size grows with the mispricing, but its position is capped: max_long shares
long, max_short shares short. Those caps are "limits to arbitrage": the trader
knows the price is wrong but has limited capital to correct it. Shorting is
much harder than buying (on BIST it is restricted, and in a low-float stock
there are few shares to borrow), so max_short is small by default

Orders are passive limit orders at the reference price, re-posted every time
it acts (the previous one is cancelled first)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from contagionsim.market.market import Market
from contagionsim.orderbook import Side


@dataclass(frozen=True, slots=True)
class ValueTraderParams:
    fundamental: int = 1000  # ticks
    threshold: float = 0.05  # ignore mispricing smaller than 5%
    p_act: float = 0.3
    shares_per_pct: int = 20  # order size per 1% of mispricing
    max_long: int = 5000  # shares
    max_short: int = 500  # shares (short-sale constraint)


class ValueTrader:
    def __init__(self, name: str, params: ValueTraderParams | None = None) -> None:
        self.name = name
        self.params = params or ValueTraderParams()
        self.order_id: int | None = None

    def act(self, t: int, market: Market, rng: np.random.Generator) -> None:
        p = self.params
        if rng.random() >= p.p_act:
            return
        if self.order_id is not None:
            market.cancel(self.name, self.order_id)
            self.order_id = None

        ref = market.reference_price()
        mispricing = (ref - p.fundamental) / p.fundamental
        if abs(mispricing) < p.threshold:
            return

        position = market.accounts[self.name].position
        size = int(p.shares_per_pct * abs(mispricing) * 100)
        if mispricing > 0:
            side, room, price = Side.SELL, p.max_short + position, math.ceil(ref)
        else:
            side, room, price = Side.BUY, p.max_long - position, math.floor(ref)
        size = min(size, room)
        if size <= 0 or price < 1:
            return  # at its capacity limit: knows the price is wrong, cannot act

        result = market.submit_limit(self.name, side, size, price)
        if result.resting > 0:
            self.order_id = result.order_id
