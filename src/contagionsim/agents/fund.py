"""Investment fund concentrated in one illiquid stock

State: cash and shares (its Market account), units held by investors,
and pending redemptions (cash owed to investors but not yet paid)

    NAV = cash - pending + shares * mark price
    NAV per unit = NAV / units

Each step, in this order:
    1. cancel last step's unfilled buy order
    2. investor flows (every flow_interval steps): performance chasing,
       flow = alpha + beta * (return over the lookback) + noise, as a fraction of NAV
    3. scheduled shocks: a one-off redemption, or a new position cap
    4. pay redemptions from cash; if cash is short, SELL shares (forced seller)
    5. if above the cap, SELL the excess (forced seller)
    6. otherwise, invest spare cash in the stock up to target_weight,
       never above its own concentration limit (max_float_share) or the cap

Redeemers are paid at today's NAV, but the selling needed to pay them pushes
the price down afterwards, so the cost falls on the investors who stay.
That first-mover advantage is what turns redemptions into a run

How much to sell per step is delegated to a LiquidationStrategy
(M3: naive, dump everything now. M6: pro-rata, Almgren-Chriss)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from contagionsim.market.market import Market
from contagionsim.orderbook import Side

# Liquidation strategies


class LiquidationStrategy(Protocol):
    def shares_to_sell(self, needed: int, market: Market) -> int:
        """How many of the `needed` shares to sell this step"""
        ...


class NaiveLiquidation:
    """Sell everything needed immediately with a market order"""

    def shares_to_sell(self, needed: int, market: Market) -> int:
        return needed


# Fund


@dataclass(frozen=True, slots=True)
class FundParams:
    initial_cash: int = 5_000_000  # ticks x shares (50,000 TRY at 0.01 TRY/tick)
    target_weight: float = 0.9  # share of NAV invested in the stock
    max_buy_per_step: int = 100  # shares, so its buying is gradual
    max_slippage: int = 3  # ticks above the reference it is willing to pay
    free_float: int = 100_000  # shares available to trade, for the cap rule
    max_float_share: float = 0.25  # fund's own concentration limit (share of free float)
    # Investor flows (performance chasing)
    flow_interval: int = 100  # steps between flow events (think: one day)
    flow_lookback: int = 5  # in flow intervals
    flow_alpha: float = 0.02  # baseline inflow per interval (a popular fund)
    flow_beta: float = 2.0  # flow sensitivity to past return (performance chasing)
    flow_sigma: float = 0.01
    max_flow: float = 0.10  # cap on |flow| as a fraction of NAV per event


@dataclass(frozen=True, slots=True)
class FundShock:
    t: int
    redeem_fraction: float = 0.0  # share of units redeemed at once
    cap_fraction: float | None = None  # new cap on holdings, share of free float


@dataclass
class FundRecord:
    t: int
    nav_per_unit: float
    shares: int
    cash: int
    pending: float
    flow: float  # fraction of NAV, + subscriptions, - redemptions
    shares_sold: int


class Fund:
    def __init__(
        self,
        name: str,
        params: FundParams | None = None,
        shocks: list[FundShock] | None = None,
        liquidation: LiquidationStrategy | None = None,
    ) -> None:
        self.name = name
        self.params = params or FundParams()
        self.shocks: dict[int, list[FundShock]] = {}
        for shock in shocks or []:
            self.shocks.setdefault(shock.t, []).append(shock)
        self.liquidation = liquidation or NaiveLiquidation()
        self.units = 0.0
        self.pending = 0.0  # redemptions owed, in ticks x shares
        self.cap_shares: int | None = None
        self.order_id: int | None = None
        self.records: list[FundRecord] = []
        self.failed_at: int | None = None  # step at which it could no longer pay redeemers
        self._nav_history: list[float] = []  # NAV per unit at each flow event
        self._initialized = False

    # Valuation
    def nav(self, market: Market) -> float:
        acc = market.accounts[self.name]
        return acc.cash - self.pending + acc.position * market.reference_price()

    def nav_per_unit(self, market: Market) -> float:
        return self.nav(market) / self.units if self.units > 0 else float("nan")

    # Main step
    def act(self, t: int, market: Market, rng: np.random.Generator) -> None:
        if not self._initialized:
            self._initialize(market)
        if self.failed_at is not None:
            return  # wound up: nothing left to sell, investors take the loss
        if self.order_id is not None:
            market.cancel(self.name, self.order_id)
            self.order_id = None

        flow = 0.0
        p = self.params
        if t > 0 and t % p.flow_interval == 0:
            flow = self._investor_flows(market, rng)
            self._nav_history.append(self.nav_per_unit(market))

        for shock in self.shocks.get(t, []):
            if shock.redeem_fraction > 0:
                self._redeem_units(self.units * shock.redeem_fraction, market)
            if shock.cap_fraction is not None:
                self.cap_shares = int(shock.cap_fraction * p.free_float)

        sold = self._pay_redemptions(market)
        sold += self._comply_with_cap(market)
        if self.pending == 0 and not self._above_cap(market):
            self._invest(market)

        acc = market.accounts[self.name]
        if self.pending > 0 and acc.position == 0 and acc.cash == 0:
            self.failed_at = t
        self.records.append(
            FundRecord(
                t, self.nav_per_unit(market), acc.position, acc.cash, self.pending, flow, sold
            )
        )

    # Investors
    def _initialize(self, market: Market) -> None:
        # Launch: investors put in initial_cash at NAV per unit = 1.0 TRY (100 ticks)
        market.transfer_external(self.name, self.params.initial_cash)
        self.units = self.params.initial_cash / 100.0
        self._nav_history.append(self.nav_per_unit(market))
        self._initialized = True

    def _investor_flows(self, market: Market, rng: np.random.Generator) -> float:
        p = self.params
        # Early on there is less history than the lookback: use the oldest point
        past = self._nav_history[-min(p.flow_lookback, len(self._nav_history))]
        now = self.nav_per_unit(market)
        past_return = now / past - 1.0 if past > 0 else 0.0
        flow = p.flow_alpha + p.flow_beta * past_return + p.flow_sigma * rng.standard_normal()
        flow = float(np.clip(flow, -p.max_flow, p.max_flow))

        if flow > 0:
            cash_in = int(flow * self.nav(market))
            if cash_in > 0:
                self.units += cash_in / now
                market.transfer_external(self.name, cash_in)
        elif flow < 0:
            self._redeem_units(-flow * self.units, market)
        return flow

    def _redeem_units(self, units: float, market: Market) -> None:
        # Redeemers lock in today's NAV: the selling comes after
        units = min(units, self.units)
        self.pending += units * self.nav_per_unit(market)
        self.units -= units

    # Forced selling
    def _pay_redemptions(self, market: Market) -> int:
        sold = 0
        self._pay_from_cash(market)
        if self.pending > 0:
            acc = market.accounts[self.name]
            needed = math.ceil(self.pending / max(market.reference_price(), 1.0))
            sold = self._sell(min(needed, acc.position), market)
            self._pay_from_cash(market)
        return sold

    def _pay_from_cash(self, market: Market) -> None:
        cash = market.accounts[self.name].cash
        pay = int(min(cash, math.ceil(self.pending)))
        if pay > 0:
            market.transfer_external(self.name, -pay)
            self.pending = max(self.pending - pay, 0.0)

    def _above_cap(self, market: Market) -> bool:
        return self.cap_shares is not None and market.accounts[self.name].position > self.cap_shares

    def _comply_with_cap(self, market: Market) -> int:
        if not self._above_cap(market):
            return 0
        assert self.cap_shares is not None
        excess = market.accounts[self.name].position - self.cap_shares
        return self._sell(excess, market)

    def _sell(self, needed: int, market: Market) -> int:
        qty = min(self.liquidation.shares_to_sell(needed, market), needed)
        if qty <= 0:
            return 0
        return market.submit_market(self.name, Side.SELL, qty).filled

    # Investing
    def _invest(self, market: Market) -> None:
        p = self.params
        acc = market.accounts[self.name]
        ref = market.reference_price()
        target_value = p.target_weight * self.nav(market)
        gap_value = target_value - acc.position * ref
        if gap_value <= 0:
            return
        limit = math.ceil(ref) + p.max_slippage
        qty = min(int(gap_value / limit), int(acc.cash / limit), p.max_buy_per_step)
        limit_shares = int(p.max_float_share * p.free_float)
        if self.cap_shares is not None:
            limit_shares = min(limit_shares, self.cap_shares)
        qty = min(qty, limit_shares - acc.position)
        if qty <= 0:
            return
        # Marketable limit order: buys now, but never above `limit`, so cash cannot go negative
        result = market.submit_limit(self.name, Side.BUY, qty, limit)
        if result.resting > 0:
            self.order_id = result.order_id
