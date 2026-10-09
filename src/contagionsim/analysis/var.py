"""Value-at-Risk models and the standard backtests (Q3)

Convention: VaR is a positive number, the loss threshold. A 99% VaR of 0.02 means
"with 99% confidence the return will not be below -2%". An exception (a "hit")
is a period where the return fell below -VaR

Models, from simplest to the bank workhorse:
    historical_var : empirical quantile of the last `window` returns
    normal_var     : mean and std of the last `window` returns, normal quantile
    GARCH(1,1)     : sigma2_t = omega + alpha * e_{t-1}^2 + beta * sigma2_{t-1}
                     fitted by maximum likelihood (normal errors) on a calm window,
                     then run forward: parameters fixed, variance updated with
                     every new return (how a risk desk uses it day to day).
                     Variance targeting: omega = sample variance * (1 - alpha - beta),
                     so the long-run variance always equals the sample variance.
                     Without it, data with little volatility clustering gives
                     degenerate fits (alpha ~ 0, beta ~ 1, omega ~ 0) whose long-run
                     variance is near zero: a VaR far too small (found in Q3 testing)

Backtests (Basel-style):
    kupiec_pof        : is the NUMBER of exceptions consistent with 1 - level?
    christoffersen_ind: are exceptions INDEPENDENT, or do they cluster?
    conditional coverage = both together (chi-square with 2 degrees of freedom)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats
from scipy.special import xlogy

FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Simple models
# ---------------------------------------------------------------------------


def historical_var(returns: FloatArray, window: int, level: float = 0.99) -> FloatArray:
    """VaR for each t >= window, using returns[t - window : t] only (no look-ahead)"""
    r = np.asarray(returns, dtype=float)
    out = np.full(len(r), np.nan)
    for t in range(window, len(r)):
        out[t] = -np.quantile(r[t - window : t], 1 - level)
    return out


def normal_var(returns: FloatArray, window: int, level: float = 0.99) -> FloatArray:
    """VaR for each t >= window from a normal fitted to the previous `window` returns"""
    r = np.asarray(returns, dtype=float)
    z = stats.norm.ppf(1 - level)
    out = np.full(len(r), np.nan)
    for t in range(window, len(r)):
        past = r[t - window : t]
        out[t] = -(past.mean() + z * past.std(ddof=1))
    return out


# ---------------------------------------------------------------------------
# GARCH(1,1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Garch11:
    mu: float
    omega: float
    alpha: float
    beta: float
    sigma2_0: float | None = None  # starting variance; None = long-run variance

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_variance(self) -> float:
        return self.omega / (1.0 - self.persistence)

    def variances(self, returns: FloatArray, sigma2_0: float | None = None) -> FloatArray:
        """sigma2[t] = variance forecast for returns[t], using information up to t - 1

        len(result) == len(returns) + 1: the last value is the forecast for the next period
        """
        r = np.asarray(returns, dtype=float)
        s2 = np.empty(len(r) + 1)
        if sigma2_0 is None:
            sigma2_0 = self.long_run_variance if self.sigma2_0 is None else self.sigma2_0
        s2[0] = sigma2_0
        for t in range(len(r)):
            e = r[t] - self.mu
            s2[t + 1] = self.omega + self.alpha * e * e + self.beta * s2[t]
        return s2

    def var(self, returns: FloatArray, level: float = 0.99) -> FloatArray:
        """One-step-ahead VaR for every element of `returns` (same length)"""
        sigma = np.sqrt(self.variances(returns)[:-1])
        z = float(stats.norm.ppf(1 - level))  # scipy is untyped: make the float explicit
        return -(self.mu + z * sigma)


def _neg_loglik(theta: FloatArray, x: FloatArray, sample_var: float) -> float:
    mu, alpha, beta = theta
    if alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e10
    omega = sample_var * (1 - alpha - beta)  # variance targeting
    s2 = Garch11(mu, omega, alpha, beta).variances(x, sigma2_0=sample_var)[:-1]
    if np.any(s2 <= 0) or not np.all(np.isfinite(s2)):
        return 1e10
    e = x - mu
    return float(0.5 * np.sum(np.log(2 * np.pi) + np.log(s2) + e * e / s2))


def fit_garch11(returns: FloatArray) -> Garch11:
    """Maximum likelihood fit with normal errors and variance targeting

    Returns are standardized first (divided by their std) so the optimizer works on
    numbers of order 1, then omega and mu are scaled back. Several starting points
    guard against a bad local optimum. The fitted model starts its variance path at
    the sample variance, exactly as during estimation
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < 30:
        raise ValueError("need at least 30 returns to fit a GARCH(1,1)")
    scale = float(r.std())
    if scale == 0.0:
        raise ValueError("returns have zero variance")
    x = r / scale
    sample_var = float(np.var(x))

    best: optimize.OptimizeResult | None = None
    for alpha0, beta0 in ((0.05, 0.90), (0.10, 0.80), (0.20, 0.50), (0.02, 0.97)):
        start = np.array([x.mean(), alpha0, beta0])
        res = optimize.minimize(
            _neg_loglik,
            start,
            args=(x, sample_var),
            method="Nelder-Mead",
            options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-8},
        )
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    mu, alpha, beta = best.x
    omega = sample_var * (1 - alpha - beta)
    s2 = scale**2
    return Garch11(mu * scale, omega * s2, alpha, beta, sigma2_0=sample_var * s2)


# ---------------------------------------------------------------------------
# Backtests
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Backtest:
    n: int  # periods tested
    exceptions: int
    expected: float  # n * (1 - level)
    kupiec_lr: float
    kupiec_p: float
    christoffersen_lr: float
    christoffersen_p: float

    @property
    def cc_p(self) -> float:
        """Conditional coverage: count and independence jointly, chi-square(2)"""
        return float(stats.chi2.sf(self.kupiec_lr + self.christoffersen_lr, df=2))

    def rejected(self, alpha: float = 0.05) -> bool:
        return self.cc_p < alpha


def kupiec_pof(n: int, x: int, p: float) -> tuple[float, float]:
    """Proportion-of-failures likelihood ratio: (LR, p-value), chi-square(1)"""
    if not 0 <= x <= n or n == 0:
        raise ValueError("need 0 <= x <= n and n > 0")
    phat = x / n
    log_null = xlogy(n - x, 1 - p) + xlogy(x, p)
    log_alt = xlogy(n - x, 1 - phat) + xlogy(x, phat)
    lr = max(-2.0 * (log_null - log_alt), 0.0)
    return float(lr), float(stats.chi2.sf(lr, df=1))


def transition_counts(hits: NDArray[np.bool_]) -> tuple[int, int, int, int]:
    """(n00, n01, n10, n11): n01 = number of times no-hit was followed by a hit, etc"""
    h = np.asarray(hits, dtype=bool)
    prev, nxt = h[:-1], h[1:]
    return (
        int(np.sum(~prev & ~nxt)),
        int(np.sum(~prev & nxt)),
        int(np.sum(prev & ~nxt)),
        int(np.sum(prev & nxt)),
    )


def christoffersen_ind(n00: int, n01: int, n10: int, n11: int) -> tuple[float, float]:
    """Independence likelihood ratio: (LR, p-value), chi-square(1)

    Compares P(hit | hit yesterday) with P(hit | no hit yesterday). If hits cluster,
    the first is much larger, and the model is too slow to react to new volatility
    """
    total = n00 + n01 + n10 + n11
    if total == 0:
        return 0.0, 1.0
    pi = (n01 + n11) / total
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    log_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    log_alt = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = max(-2.0 * (log_null - log_alt), 0.0)
    return float(lr), float(stats.chi2.sf(lr, df=1))


def hits(returns: FloatArray, var: FloatArray) -> NDArray[np.bool_]:
    """True where the return fell below -VaR"""
    return np.asarray(returns, dtype=float) < -np.asarray(var, dtype=float)


def backtest(hit_series: list[NDArray[np.bool_]], level: float = 0.99) -> Backtest:
    """Pool several independent hit series (e.g. one per seed) into one backtest

    Counts are summed; transitions are counted inside each series only, never
    across the boundary between two series
    """
    p = 1 - level
    n = sum(len(h) for h in hit_series)
    x = sum(int(np.sum(h)) for h in hit_series)
    counts = np.sum([transition_counts(h) for h in hit_series], axis=0)
    k_lr, k_p = kupiec_pof(n, x, p)
    c_lr, c_p = christoffersen_ind(*(int(c) for c in counts))
    return Backtest(n, x, n * p, k_lr, k_p, c_lr, c_p)
