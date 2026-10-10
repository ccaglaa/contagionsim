"""Early-warning features and labels from public market data only (Q4)

Features at step t use the market history up to and including t, nothing later.
They are what any market participant could see: prices, quotes, depth imbalance,
volume, trading halts. Nothing from inside the fund (its pending redemptions are
private), so the model has to infer the stress from its footprint in the market

A second set, FLOW_FEATURES, adds what a regulator or the fund platform (TEFAS) sees
and the market does not: the fund's redemptions and sales as they happen. Comparing
the two sets measures how much that private information is worth for early warning

Label at step t: does the 5% index circuit breaker fire in (t, t + horizon]?

Samples are taken every `every` steps from `start`, skipping:
    - steps while the index is halted (no continuous market to read)
    - the rest of a day after its 5% breaker fired (the stage cannot fire twice a day,
      so those steps would be trivially negative and inflate the scores)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from contagionsim.agents import PortfolioFund
from contagionsim.market import ExchangeSnapshot

FloatArray = NDArray[np.float64]

FEATURES = (
    "index_ret_day",  # index vs previous close: distance to the 5% breaker
    "index_ret_50",  # index return over the last 50 steps
    "index_ret_200",
    "index_dd_from_day_high",
    "index_vol_100",  # std of 1-step index returns over the last 100 steps
    "liquid_ret_day",
    "corner_ret_day",
    "corner_near_floor",  # last trade within 1% of the daily limit-down price
    "corner_halted",
    "halts_today",  # stock circuit breakers so far today, all stocks
    "liquid_spread_50",  # mean bid-ask spread of the liquid stocks, last 50 steps (ticks)
    "corner_spread_50",
    "liquid_imbalance_20",  # mean top-of-book imbalance of the liquid stocks, last 20 steps
    "liquid_volume_ratio",  # volume over the last 50 steps vs the 50-step average so far
    "time_of_day",
)


FLOW_FEATURES = (
    "fund_net_outflow_today",  # units redeemed net of subscriptions since the open, share
    "fund_net_outflow_300",  # ... over the last 300 steps
    "fund_sold_today",  # value the fund sold since the open, share of NAV at the open
    "fund_owed_now",  # redemptions owed but not yet paid (incl. gated), share of NAV
)


def _ffill(x: FloatArray, fill: float) -> FloatArray:
    """Forward-fill NaNs (a halted book has no quotes), initial NaNs -> fill"""
    out: FloatArray = np.array(x, dtype=np.float64)
    last = fill
    for i, v in enumerate(out):
        if math.isnan(v):
            out[i] = last
        else:
            last = v
    return out


def _trailing_mean(x: FloatArray, window: int) -> FloatArray:
    """Mean of x[max(0, t - window + 1) : t + 1] for every t (uses only the past)"""
    c = np.cumsum(np.r_[0.0, x])
    t = np.arange(len(x))
    lo = np.maximum(0, t - window + 1)
    return np.asarray((c[t + 1] - c[lo]) / (t + 1 - lo), dtype=np.float64)


def _trailing_std(x: FloatArray, window: int) -> FloatArray:
    m = _trailing_mean(x, window)
    m2 = _trailing_mean(x * x, window)
    return np.asarray(np.sqrt(np.maximum(m2 - m * m, 0.0)), dtype=np.float64)


def _lagged_return(p: FloatArray, k: int) -> FloatArray:
    past = np.r_[np.full(k, p[0]), p[:-k]]
    return np.asarray(p / past - 1.0, dtype=np.float64)


def run_features(
    history: Sequence[ExchangeSnapshot],
    stock_halt_steps: Sequence[int],
    concentrated: str,
    liquid: Sequence[str],
    concentrated_limit: float,
    steps_per_day: int = 1000,
) -> FloatArray:
    """Feature matrix, one row per step of `history`, columns in FEATURES order"""
    n = len(history)
    t = np.arange(n)
    day = t // steps_per_day
    index = np.array([s.index for s in history])

    def previous_close(p: FloatArray) -> FloatArray:
        # close of the previous day; on day 0, the first price
        days = [int(d) for d in day]
        closes = {d: float(p[min((d + 1) * steps_per_day - 1, n - 1)]) for d in range(days[-1] + 1)}
        return np.array([closes[d - 1] if d > 0 else float(p[0]) for d in days], dtype=np.float64)

    liq_price = np.array([np.mean([s.assets[x].last_price for x in liquid]) for s in history])
    corner = np.array([s.assets[concentrated].last_price for s in history], dtype=np.float64)
    corner_base = previous_close(corner)
    floor = np.ceil(corner_base * (1 - concentrated_limit))

    day_high = np.empty(n)
    for i in range(n):
        start = day[i] * steps_per_day
        day_high[i] = index[i] if i == start else max(day_high[i - 1], index[i])

    halts = np.zeros(n)
    for h in stock_halt_steps:
        if 0 <= h < n:
            end = min((h // steps_per_day + 1) * steps_per_day, n)
            halts[h:end] += 1  # counts from the halt to the end of that day

    def spread(sym: str) -> FloatArray:
        raw = []
        for s in history:
            ask, bid = s.assets[sym].best_ask, s.assets[sym].best_bid
            raw.append(float(ask - bid) if ask is not None and bid is not None else np.nan)
        return _ffill(np.array(raw, dtype=np.float64), 2.0)

    liq_spread = np.mean([spread(x) for x in liquid], axis=0)
    imb_raw = np.array(
        [
            np.mean(
                [
                    s.assets[x].imbalance if s.assets[x].imbalance is not None else np.nan
                    for x in liquid
                ]
            )  # fmt: skip
            for s in history
        ],
        dtype=np.float64,
    )
    volume = np.array([sum(s.assets[x].volume for x in liquid) for s in history], dtype=np.float64)
    vol50 = _trailing_mean(volume, 50) * 50
    avg50 = np.cumsum(volume) / (t + 1) * 50

    index_ret_1 = np.r_[0.0, np.diff(np.log(index))]
    columns = {
        "index_ret_day": index / previous_close(index) - 1.0,
        "index_ret_50": _lagged_return(index, 50),
        "index_ret_200": _lagged_return(index, 200),
        "index_dd_from_day_high": index / day_high - 1.0,
        "index_vol_100": _trailing_std(index_ret_1, 100),
        "liquid_ret_day": liq_price / previous_close(liq_price) - 1.0,
        "corner_ret_day": corner / corner_base - 1.0,
        "corner_near_floor": (corner <= floor * 1.01).astype(float),
        "corner_halted": np.array([float(s.halted[concentrated]) for s in history]),
        "halts_today": halts,
        "liquid_spread_50": _trailing_mean(liq_spread, 50),
        "corner_spread_50": _trailing_mean(spread(concentrated), 50),
        "liquid_imbalance_20": _trailing_mean(_ffill(imb_raw, 0.0), 20),
        "liquid_volume_ratio": vol50 / np.maximum(avg50, 1e-9),
        "time_of_day": (t % steps_per_day) / steps_per_day,
    }
    return np.column_stack([columns[f] for f in FEATURES])


def fund_flow_features(fund: PortfolioFund, steps_per_day: int = 1000) -> FloatArray:
    """FLOW_FEATURES for every step of the fund's records (uses only the past)"""
    units = np.asarray(fund.units_history, dtype=np.float64)
    n = len(units)
    t = np.arange(n)
    day_start = np.maximum(t - t % steps_per_day - 1, 0)  # last step of the previous day
    nav_unit = np.array([r.nav_per_unit for r in fund.records], dtype=np.float64)
    nav_unit = np.where(np.isfinite(nav_unit), nav_unit, 0.0)
    nav = nav_unit * units
    proceeds = np.array([sum(r.proceeds.values()) for r in fund.records], dtype=np.float64)
    sold_today = np.empty(n)
    for i in range(n):
        sold_today[i] = proceeds[i] if i % steps_per_day == 0 else sold_today[i - 1] + proceeds[i]
    owed = np.array([r.pending + r.gated for r in fund.records], dtype=np.float64)
    lag = np.r_[np.full(300, units[0]), units[:-300]] if n > 300 else np.full(n, units[0])

    def share(x: FloatArray, base: FloatArray) -> FloatArray:
        return np.where(base > 0, x / np.where(base > 0, base, 1.0), 0.0)

    return np.column_stack(
        [
            share(units[day_start] - units, units[day_start]),
            share(lag - units, lag),
            share(sold_today, nav[day_start]),
            share(owed, nav + owed),
        ]
    )


def run_labels(n: int, breaker_steps: Sequence[int], horizon: int) -> NDArray[np.bool_]:
    """label[t] = a 5% index breaker fires in (t, t + horizon]"""
    y = np.zeros(n, dtype=bool)
    for b in breaker_steps:
        y[max(0, b - horizon) : b] = True
    return y


def sample_steps(
    history: Sequence[ExchangeSnapshot],
    breaker_steps: Sequence[int],
    start: int,
    every: int,
    steps_per_day: int = 1000,
) -> NDArray[np.int64]:
    """Steps used as samples: every `every` from `start`, minus halts and post-breaker days"""
    n = len(history)
    keep = np.zeros(n, dtype=bool)
    keep[start::every] = True
    keep &= ~np.array([s.index_halted for s in history])
    for b in breaker_steps:
        keep[b : min((b // steps_per_day + 1) * steps_per_day, n)] = False
    return np.flatnonzero(keep)
