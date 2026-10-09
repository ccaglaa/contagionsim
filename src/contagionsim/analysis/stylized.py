"""Stylized facts of returns (Cont 2001): checks that a simulated price looks real

    1. Heavy tails: excess kurtosis > 0 (a normal distribution has 0)
    2. No linear predictability: autocorrelation of returns close to 0
    3. Volatility clustering: autocorrelation of |returns| positive and slowly decaying

Used in M2 as a sanity check of the baseline market, and in M8 to compare with BIST data
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def log_returns(prices: FloatArray, horizon: int = 1) -> FloatArray:
    """Non-overlapping log returns over `horizon` steps"""
    sampled = np.asarray(prices, dtype=float)[::horizon]
    return np.diff(np.log(sampled))


def excess_kurtosis(x: FloatArray) -> float:
    x = np.asarray(x, dtype=float)
    d = x - x.mean()
    var = float(np.mean(d**2))
    if var == 0.0:
        return float("nan")
    return float(np.mean(d**4) / var**2 - 3.0)


def autocorr(x: FloatArray, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    if lag <= 0 or lag >= len(x):
        raise ValueError("lag must be in [1, len(x) - 1]")
    a, b = x[:-lag], x[lag:]
    if a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@dataclass(frozen=True, slots=True)
class StylizedFacts:
    horizon: int
    n_returns: int
    std: float
    excess_kurtosis: float
    acf_returns_lag1: float
    acf_abs_returns_lag1: float
    acf_abs_returns_lag10: float

    def summary(self) -> str:
        return (
            f"horizon {self.horizon} steps, n={self.n_returns}\n"
            f"  std of returns          {self.std * 1e4:8.1f} bp\n"
            f"  excess kurtosis         {self.excess_kurtosis:8.2f}   (normal = 0, real > 0)\n"
            f"  acf(r, 1)               {self.acf_returns_lag1:8.3f}   (real ~ 0)\n"
            f"  acf(|r|, 1)             {self.acf_abs_returns_lag1:8.3f}   (real > 0)\n"
            f"  acf(|r|, 10)            {self.acf_abs_returns_lag10:8.3f}   (real > 0, decaying)"
        )


def stylized_facts(prices: FloatArray, horizon: int = 1) -> StylizedFacts:
    r = log_returns(prices, horizon)
    a = np.abs(r)
    return StylizedFacts(
        horizon=horizon,
        n_returns=len(r),
        std=float(r.std()),
        excess_kurtosis=excess_kurtosis(r),
        acf_returns_lag1=autocorr(r, 1),
        acf_abs_returns_lag1=autocorr(a, 1),
        acf_abs_returns_lag10=autocorr(a, 10),
    )
