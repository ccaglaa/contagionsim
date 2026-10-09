"""Noise trader: random order flow with no view on value

Each step, with probability p_act it sends one order:
    - with probability p_market a market order (consumes liquidity)
    - otherwise a limit order some ticks away from the reference price (adds liquidity)
Side is a fair coin
Sizes are Pareto (power-law) distributed: many small orders, a few very large
ones, as in real order flow. This is what gives returns their fat tails
Limit order distance from the reference price is geometric, at least 1 tick
Each of its resting orders is cancelled with probability p_cancel per step,
so stale orders do not pile up forever
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from contagionsim.market.market import Market
from contagionsim.orderbook import Side


@dataclass(frozen=True, slots=True)
class NoiseTraderParams:
    p_act: float = 0.5
    p_market: float = 0.3
    p_cancel: float = 0.05
    min_size: int = 10  # shares, smallest order
    size_alpha: float = 1.6  # Pareto tail exponent: lower = fatter tail
    max_size: int = 5000  # cap so one draw cannot be absurd
    mean_offset: float = 4.0  # ticks away from the reference price, for limit orders


class NoiseTrader:
    def __init__(self, name: str, params: NoiseTraderParams | None = None) -> None:
        self.name = name
        self.params = params or NoiseTraderParams()
        self.live_orders: list[int] = []

    def act(self, t: int, market: Market, rng: np.random.Generator) -> None:
        p = self.params
        self._cancel_some(market, rng)
        if rng.random() >= p.p_act:
            return

        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        size = self._draw_size(rng)

        if rng.random() < p.p_market:
            market.submit_market(self.name, side, size)
            return

        # offset >= 1 tick: limit orders never cross the reference price
        offset = int(rng.geometric(1.0 / p.mean_offset))
        ref = market.reference_price()
        if side is Side.BUY:
            price = math.ceil(ref) - offset
        else:
            price = math.floor(ref) + offset
        price = max(price, 1)

        result = market.submit_limit(self.name, side, size, price)
        if result.resting > 0:
            self.live_orders.append(result.order_id)

    def _draw_size(self, rng: np.random.Generator) -> int:
        p = self.params
        # numpy pareto is the Lomax form: (1 + X) is a classic Pareto with minimum 1
        size = p.min_size * (1.0 + rng.pareto(p.size_alpha))
        return min(int(size), p.max_size)

    def _cancel_some(self, market: Market, rng: np.random.Generator) -> None:
        keep = []
        for oid in self.live_orders:
            if market.book.get_order(oid) is None:
                continue  # already fully filled, forget it
            if rng.random() < self.params.p_cancel:
                market.cancel(self.name, oid)
            else:
                keep.append(oid)
        self.live_orders = keep
