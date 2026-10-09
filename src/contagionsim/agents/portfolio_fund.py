"""Multi-asset fund: one concentrated illiquid stock + liquid index stocks (M4)

Same investor model as the single-asset Fund (M3): performance-chasing flows,
redemptions locked at today's NAV, regulatory cap shocks. The difference is
where the money goes and where it comes from:

Investing (no stress):
    1. the concentrated stock, up to its concentration limit (share of free float)
    2. everything else equally across the liquid stocks
    so the fund stays ~target_invested in stocks, not piling up cash (the M3 flaw)

Raising cash (redemptions) -- "sell what can be sold":
    the cash needed is split across holdings by value, but ONLY over stocks that
    can trade right now: not halted, not locked at limit-down. If the concentrated
    stock is locked at its floor, the whole burden falls on the liquid stocks.
    That is the contagion channel: liquid stocks fall because of who owns them

Soft close: with max_aum set, the fund stops taking subscriptions above that size
(real funds do this; without it, 10 flow events a day compound without limit)

Cap compliance: the excess of the concentrated stock is sold with market orders
every step until the fund is under the cap (it can only sell what the limit allows)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from contagionsim.agents.fund import FundShock
from contagionsim.market.exchange import Exchange
from contagionsim.market.market import Market
from contagionsim.orderbook import Side


@dataclass(frozen=True, slots=True)
class PortfolioFundParams:
    concentrated: str = "CORNER"
    liquid: tuple[str, ...] = ("L1", "L2", "L3", "L4")
    initial_cash: int = 5_000_000  # ticks x shares (50,000 TRY)
    target_invested: float = 0.95  # share of NAV in stocks
    concentrated_free_float: int = 100_000
    max_float_share: float = 0.25  # own concentration limit in the concentrated stock
    max_buy_per_step: int = 100  # shares per symbol per step
    max_slippage: int = 3  # ticks above the reference it will pay
    # Investor flows (same model as Fund)
    flow_interval: int = 100
    flow_lookback: int = 5
    flow_alpha: float = 0.02
    flow_beta: float = 2.0
    flow_sigma: float = 0.01
    max_flow: float = 0.10
    max_aum: float | None = None  # soft close: no new subscriptions above this NAV (ticks)


@dataclass
class PortfolioRecord:
    t: int
    nav_per_unit: float
    cash: int
    pending: float
    flow: float
    positions: dict[str, int]
    sold: dict[str, int] = field(default_factory=dict)  # shares sold this step


class PortfolioFund:
    def __init__(
        self,
        name: str,
        params: PortfolioFundParams | None = None,
        shocks: list[FundShock] | None = None,
    ) -> None:
        self.name = name
        self.params = params or PortfolioFundParams()
        self.shocks: dict[int, list[FundShock]] = {}
        for shock in shocks or []:
            self.shocks.setdefault(shock.t, []).append(shock)
        self.units = 0.0
        self.pending = 0.0
        self.cap_shares: int | None = None
        self.order_ids: dict[str, int] = {}
        self.records: list[PortfolioRecord] = []
        self.failed_at: int | None = None
        self._nav_history: list[float] = []
        self._initialized = False

    @property
    def symbols(self) -> tuple[str, ...]:
        return (self.params.concentrated, *self.params.liquid)

    # Valuation
    def holdings_value(self, ex: Exchange) -> dict[str, float]:
        return {s: ex.position(self.name, s) * ex[s].reference_price() for s in self.symbols}

    def nav(self, ex: Exchange) -> float:
        return ex.cash(self.name) - self.pending + sum(self.holdings_value(ex).values())

    def nav_per_unit(self, ex: Exchange) -> float:
        return self.nav(ex) / self.units if self.units > 0 else float("nan")

    # Main step
    def act(self, t: int, ex: Exchange, rng: np.random.Generator) -> None:
        if not self._initialized:
            self._initialize(ex)
        if self.failed_at is not None:
            return
        for symbol, oid in self.order_ids.items():
            ex[symbol].cancel(self.name, oid)
        self.order_ids = {}

        p = self.params
        flow = 0.0
        if t > 0 and t % p.flow_interval == 0:
            flow = self._investor_flows(ex, rng)
            self._nav_history.append(self.nav_per_unit(ex))

        for shock in self.shocks.get(t, []):
            if shock.redeem_fraction > 0:
                self._redeem_units(self.units * shock.redeem_fraction, ex)
            if shock.cap_fraction is not None:
                self.cap_shares = int(shock.cap_fraction * p.concentrated_free_float)

        sold: dict[str, int] = {}
        self._raise_cash(ex, sold)
        self._comply_with_cap(ex, sold)
        if self.pending == 0 and not self._above_cap(ex):
            self._invest(ex)

        positions = {s: ex.position(self.name, s) for s in self.symbols}
        if self.pending > 0 and ex.cash(self.name) == 0 and not any(positions.values()):
            self.failed_at = t
        self.records.append(
            PortfolioRecord(
                t, self.nav_per_unit(ex), ex.cash(self.name), self.pending, flow, positions, sold
            )
        )

    # Investors (same model as Fund)
    def _initialize(self, ex: Exchange) -> None:
        ex.transfer_external(self.name, self.params.initial_cash)
        self.units = self.params.initial_cash / 100.0  # NAV per unit starts at 1 TRY
        self._nav_history.append(self.nav_per_unit(ex))
        self._initialized = True

    def _investor_flows(self, ex: Exchange, rng: np.random.Generator) -> float:
        p = self.params
        past = self._nav_history[-min(p.flow_lookback, len(self._nav_history))]
        now = self.nav_per_unit(ex)
        past_return = now / past - 1.0 if past > 0 else 0.0
        flow = p.flow_alpha + p.flow_beta * past_return + p.flow_sigma * rng.standard_normal()
        flow = float(np.clip(flow, -p.max_flow, p.max_flow))
        if flow > 0 and p.max_aum is not None and self.nav(ex) >= p.max_aum:
            flow = 0.0  # soft-closed: popular, but no longer accepting new money
        if flow > 0:
            cash_in = int(flow * self.nav(ex))
            if cash_in > 0:
                self.units += cash_in / now
                ex.transfer_external(self.name, cash_in)
        elif flow < 0:
            self._redeem_units(-flow * self.units, ex)
        return flow

    def _redeem_units(self, units: float, ex: Exchange) -> None:
        units = min(units, self.units)
        self.pending += units * self.nav_per_unit(ex)
        self.units -= units

    # Raising cash: sell what can be sold
    def sellable(self, ex: Exchange) -> list[str]:
        """Symbols the fund holds that can trade now: not halted, not locked limit-down"""
        out = []
        for s in self.symbols:
            m = ex[s]
            if ex.position(self.name, s) > 0 and not m.halted and not _locked_down(m):
                out.append(s)
        return out

    def _raise_cash(self, ex: Exchange, sold: dict[str, int]) -> None:
        self._pay_from_cash(ex)
        shortfall = self.pending
        if shortfall <= 0:
            return
        candidates = self.sellable(ex)
        values = self.holdings_value(ex)
        total = sum(values[s] for s in candidates)
        if total <= 0:
            return  # nothing can be sold this step: wait (halt or limit-down everywhere)
        for s in candidates:
            share = shortfall * values[s] / total
            qty = min(
                math.ceil(share / max(ex[s].reference_price(), 1.0)), ex.position(self.name, s)
            )
            self._sell(ex, s, qty, sold)
        self._pay_from_cash(ex)

    def _pay_from_cash(self, ex: Exchange) -> None:
        pay = int(min(ex.cash(self.name), math.ceil(self.pending)))
        if pay > 0:
            ex.transfer_external(self.name, -pay)
            self.pending = max(self.pending - pay, 0.0)

    def _above_cap(self, ex: Exchange) -> bool:
        held = ex.position(self.name, self.params.concentrated)
        return self.cap_shares is not None and held > self.cap_shares

    def _comply_with_cap(self, ex: Exchange, sold: dict[str, int]) -> None:
        if not self._above_cap(ex):
            return
        assert self.cap_shares is not None
        s = self.params.concentrated
        self._sell(ex, s, ex.position(self.name, s) - self.cap_shares, sold)

    def _sell(self, ex: Exchange, symbol: str, qty: int, sold: dict[str, int]) -> None:
        if qty <= 0:
            return
        filled = ex[symbol].submit_market(self.name, Side.SELL, qty).filled
        if filled:
            sold[symbol] = sold.get(symbol, 0) + filled

    # Investing
    def _invest(self, ex: Exchange) -> None:
        p = self.params
        nav = self.nav(ex)
        spare = ex.cash(self.name) - (1 - p.target_invested) * nav
        if spare <= 0:
            return
        values = self.holdings_value(ex)
        # 1. concentrated stock, up to its limit
        c = p.concentrated
        limit_shares = int(p.max_float_share * p.concentrated_free_float)
        if self.cap_shares is not None:
            limit_shares = min(limit_shares, self.cap_shares)
        room = limit_shares - ex.position(self.name, c)
        spare -= self._buy(ex, c, min(room, p.max_buy_per_step), spare)
        # 2. the rest equally across liquid stocks, toward equal value targets
        invested_target = p.target_invested * nav - values[c]
        per_liquid = invested_target / len(p.liquid)
        for s in p.liquid:
            if spare <= 0:
                break
            gap = per_liquid - values[s]
            if gap <= 0:
                continue
            ref = ex[s].reference_price()
            qty = min(int(gap / (ref + p.max_slippage)), p.max_buy_per_step)
            spare -= self._buy(ex, s, qty, spare)

    def _buy(self, ex: Exchange, symbol: str, qty: int, budget: float) -> float:
        """Marketable limit buy within budget. Returns the cash committed (upper bound)"""
        m = ex[symbol]
        if qty <= 0 or m.halted:
            return 0.0
        limit = min(math.ceil(m.reference_price()) + self.params.max_slippage, m.ceiling)
        qty = min(qty, int(budget / limit))
        if qty <= 0:
            return 0.0
        result = m.submit_limit(self.name, Side.BUY, qty, limit)
        if result.resting > 0:
            self.order_ids[symbol] = result.order_id
        return float(qty * limit)


def _locked_down(market: Market) -> bool:
    """Limit-down: last trade at the floor and no bid above it"""
    if market.rules is None:
        return False
    bid = market.book.best_bid()
    return market.last_price <= market.floor and (bid is None or bid <= market.floor)
