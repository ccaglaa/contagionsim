"""Contagion metrics for one shocked mini-BIST run

All drawdowns are measured from the last step before the shock (t_shock - 1) to the
lowest point after it, so they are <= 0. "Liquid" is the equal-weighted average
price path of the liquid stocks: the stocks with nothing wrong with them
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from contagionsim.agents import PortfolioFund
from contagionsim.market import ExchangeSimulation
from contagionsim.scenarios import FUNDAMENTAL


@dataclass(frozen=True, slots=True)
class ContagionMetrics:
    concentrated_drawdown: float
    liquid_drawdown: float
    index_drawdown: float
    index_breaker_5: bool  # the 5% index circuit breaker fired after the shock
    index_breaker_7: bool
    nav_drawdown: float  # fund NAV per unit, for the investors who stayed: worst point
    nav_end_change: float  # ... and at the end of the run
    execution_discount: float  # 1 - proceeds / (shares sold x pre-shock price)
    # (value of shares sold at fair value - cash received) / NAV before the shock.
    # Positive = the fund sold below what the shares were worth. Unlike NAV, it cannot
    # be flattered by marking unsold shares at an inflated market price
    sale_loss_vs_value: float
    concentrated_weight: float  # concentrated stock's share of the fund's stock holdings (value)
    concentrated_share_of_sales: float  # its share of the value sold after the shock
    owed_after_1_day: float  # redemptions still unpaid one day after the shock, % of NAV
    fund_failed: bool
    concentrated_held: int  # shares of the concentrated stock just before the shock

    def as_dict(self) -> dict[str, float | bool | int]:
        return asdict(self)


def _drawdown(path: NDArray[np.float64], t: int) -> float:
    before = float(path[t - 1])
    return float(np.min(path[t:])) / before - 1.0 if before > 0 else float("nan")


def contagion_metrics(
    sim: ExchangeSimulation,
    fund: PortfolioFund,
    t_shock: int,
    fair_values: dict[str, float] | None = None,
) -> ContagionMetrics:
    """fair_values: ticks per share after the shock (default: every stock at FUNDAMENTAL)"""
    if not 0 < t_shock < len(sim.history):
        raise ValueError("t_shock must fall inside the simulated history")
    h = sim.history
    c, liquid = fund.params.concentrated, fund.params.liquid
    corner = np.array([s.assets[c].last_price for s in h], dtype=float)
    liq = np.array([np.mean([s.assets[x].last_price for x in liquid]) for s in h])
    index = np.array([s.index for s in h])
    nav = np.array([r.nav_per_unit for r in fund.records])

    breakers = {th for t, th in sim.exchange.index_halts if t >= t_shock}

    # Value weights just before the shock, and value actually sold after it.
    # If the concentrated stock's share of sales is below its weight, halts and
    # limit-down diverted the selling onto the liquid stocks
    before = fund.records[t_shock - 1].positions
    prices = {s: h[t_shock - 1].assets[s].last_price for s in (c, *liquid)}
    held_value = {s: before[s] * prices[s] for s in prices}
    total_held = sum(held_value.values())
    sold_value = {s: 0.0 for s in prices}
    for r, snap in zip(fund.records[t_shock:], h[t_shock:], strict=True):
        for s, q in r.sold.items():
            sold_value[s] += q * snap.assets[s].last_price
    total_sold = sum(sold_value.values())
    pre_value = sum(r.sold.get(s, 0) * prices[s] for r in fund.records[t_shock:] for s in prices)
    received = sum(sum(r.proceeds.values()) for r in fund.records[t_shock:])
    fair = {s: float(FUNDAMENTAL) for s in prices}
    fair.update(fair_values or {})
    fair_value_sold = sum(r.sold.get(s, 0) * fair[s] for r in fund.records[t_shock:] for s in fair)
    nav_before = fund.records[t_shock - 1].nav_per_unit * fund.units_history[t_shock - 1]
    one_day = min(t_shock + 1000, len(fund.records) - 1)
    owed = fund.records[one_day].pending + fund.records[one_day].gated

    return ContagionMetrics(
        concentrated_drawdown=_drawdown(corner, t_shock),
        liquid_drawdown=_drawdown(liq, t_shock),
        index_drawdown=_drawdown(index, t_shock),
        index_breaker_5=0.05 in breakers,
        index_breaker_7=0.07 in breakers,
        nav_drawdown=float(np.nanmin(nav[t_shock:]) / nav[t_shock - 1] - 1.0),
        nav_end_change=float(nav[-1] / nav[t_shock - 1] - 1.0),
        execution_discount=1.0 - received / pre_value if pre_value else 0.0,
        sale_loss_vs_value=(fair_value_sold - received) / nav_before if nav_before else 0.0,
        concentrated_weight=held_value[c] / total_held if total_held else 0.0,
        concentrated_share_of_sales=sold_value[c] / total_sold if total_sold else 0.0,
        owed_after_1_day=owed / nav_before if nav_before else 0.0,
        fund_failed=fund.failed_at is not None,
        concentrated_held=fund.records[t_shock - 1].positions[c],
    )
