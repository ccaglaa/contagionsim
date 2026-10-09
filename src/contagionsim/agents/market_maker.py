"""Market maker: quotes both sides, earns the spread, manages inventory risk

Every step it cancels its old quotes and posts a new bid and ask around a
reservation price (a simplified Avellaneda-Stoikov rule):

    reservation = reference - skew * inventory
    half_spread = base_half_spread + vol_mult * sigma

    - Long inventory -> reservation below the mid -> cheaper ask, lower bid:
      it sells more easily and buys less, pushing inventory back to 0
    - Higher volatility -> wider spread: compensation for riskier inventory
    - At |inventory| >= max_inventory it stops quoting the side that would grow it

sigma is an EWMA estimate of the per-step change in the reference price (ticks)
Quotes are post-only: never priced through the opposite side of the book,
because a market maker provides liquidity, it does not take it
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from contagionsim.market.market import Market
from contagionsim.orderbook import Side


@dataclass(frozen=True, slots=True)
class MarketMakerParams:
    base_half_spread: float = 1.0  # ticks
    vol_mult: float = 1.0
    skew: float = 0.005  # ticks of reservation shift per share of inventory
    quote_size: int = 100
    max_inventory: int = 2000
    ewma_lambda: float = 0.94  # RiskMetrics decay


class MarketMaker:
    def __init__(self, name: str, params: MarketMakerParams | None = None) -> None:
        self.name = name
        self.params = params or MarketMakerParams()
        self.quote_ids: list[int] = []
        self.sigma2 = 1.0  # variance of reference price changes, ticks^2
        self._prev_ref: float | None = None

    def quotes(self, reference: float, inventory: int) -> tuple[int | None, int | None]:
        """Pure pricing rule: (bid, ask) in ticks, None for a side not quoted"""
        p = self.params
        reservation = reference - p.skew * inventory
        half = p.base_half_spread + p.vol_mult * math.sqrt(self.sigma2)
        bid: int | None = max(math.floor(reservation - half), 1)
        ask: int | None = max(math.ceil(reservation + half), 2)
        if inventory >= p.max_inventory:
            bid = None
        if inventory <= -p.max_inventory:
            ask = None
        return bid, ask

    def act(self, t: int, market: Market, rng: np.random.Generator) -> None:
        for oid in self.quote_ids:
            market.cancel(self.name, oid)  # False if already filled, that is fine
        self.quote_ids = []

        ref = market.reference_price()
        self._update_vol(ref)
        inventory = market.accounts[self.name].position
        bid, ask = self.quotes(ref, inventory)

        # Post-only: stay strictly behind the other side of the book
        book = market.book
        best_ask, best_bid = book.best_ask(), book.best_bid()
        if bid is not None and best_ask is not None:
            bid = min(bid, best_ask - 1)
        if ask is not None and best_bid is not None:
            ask = max(ask, best_bid + 1)

        size = self.params.quote_size
        if bid is not None and bid >= 1:
            self.quote_ids.append(market.submit_limit(self.name, Side.BUY, size, bid).order_id)
        if ask is not None:
            self.quote_ids.append(market.submit_limit(self.name, Side.SELL, size, ask).order_id)

    def _update_vol(self, ref: float) -> None:
        if self._prev_ref is not None:
            lam = self.params.ewma_lambda
            change = ref - self._prev_ref
            self.sigma2 = lam * self.sigma2 + (1 - lam) * change * change
        self._prev_ref = ref
