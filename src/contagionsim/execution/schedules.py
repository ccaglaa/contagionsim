"""Liquidation schedules: how fast a forced seller works through what it must sell (Q2)

A schedule is a function F(elapsed) = share of the sell programme that should be
done `elapsed` steps after it started (0 <= F <= 1, F = 1 at the horizon)

    Immediate     F = 1 at once: dump everything now (the naive fund of M3/M4)
    Linear (TWAP) F grows in a straight line over the horizon
    AlmgrenChriss F follows the optimal trajectory of Almgren & Chriss (2000):
                      holdings x(u) = X * sinh(k (1 - u)) / sinh(k),  u = elapsed / horizon
                  so F(u) = 1 - sinh(k (1 - u)) / sinh(k)
                  k = kappa * T is the urgency. In the paper kappa = sqrt(lambda sigma^2 / eta)
                  (lambda: risk aversion, sigma: volatility, eta: temporary impact per share).
                  k -> 0 gives back the straight line (TWAP); large k front-loads the
                  selling and tends to Immediate. One family, one knob

The schedule says WHEN to sell; the fund still decides WHAT (its "sell what can be sold"
rule), and anything the market would not take (halt, limit-down) is caught up later
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


class Schedule(Protocol):
    def fraction_done(self, elapsed: int) -> float:
        """Share of the programme to have completed after `elapsed` steps (step 0 included)"""
        ...


@dataclass(frozen=True, slots=True)
class Immediate:
    def fraction_done(self, elapsed: int) -> float:
        return 1.0


@dataclass(frozen=True, slots=True)
class Linear:
    horizon: int  # steps

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1 step")

    def fraction_done(self, elapsed: int) -> float:
        # +1: the step on which the programme starts already sells its share
        return min(1.0, (elapsed + 1) / self.horizon)


@dataclass(frozen=True, slots=True)
class AlmgrenChriss:
    horizon: int  # steps
    urgency: float  # k = kappa * T, dimensionless

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1 step")
        if self.urgency < 0:
            raise ValueError("urgency must be >= 0")

    def fraction_done(self, elapsed: int) -> float:
        u = min(1.0, (elapsed + 1) / self.horizon)
        k = self.urgency
        if k < 1e-6:
            return u  # limit k -> 0: straight line (TWAP)
        return 1.0 - math.sinh(k * (1.0 - u)) / math.sinh(k)


def kappa_from_model(risk_aversion: float, sigma: float, eta: float) -> float:
    """Almgren-Chriss kappa = sqrt(lambda * sigma^2 / eta), per step

    Multiply by the horizon (in steps) to get the dimensionless urgency k
    """
    if risk_aversion < 0 or sigma < 0 or eta <= 0:
        raise ValueError("need risk_aversion >= 0, sigma >= 0, eta > 0")
    return math.sqrt(risk_aversion * sigma * sigma / eta)
