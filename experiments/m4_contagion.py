"""M4: fire-sale contagion from a cornered stock to the whole index

Mini BIST: CORNER (Ana Pazar, +/-15%, breaker 7.5%) + four liquid large caps
(Yildiz Pazar, +/-20%, breaker 10%), cap-weighted index with the 5% / 7% breaker.
The fund holds 25% of CORNER's float and the rest of its money in the liquid stocks.

Two runs, same seed, identical until the shock (day 8, step 50):
    calm    : nothing happens
    shocked : CORNER cap cut to 8% of free float + 30% of fund units redeemed

Usage:
    python experiments/m4_contagion.py
    python experiments/m4_contagion.py --seed 2 --redeem 0.5 --days 14

Prints a summary and saves results/m4_contagion.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from contagionsim.agents import FundShock, PortfolioFund
from contagionsim.market import ExchangeSimulation
from contagionsim.scenarios import FUNDAMENTAL, mini_bist

TICK = 0.01
STEPS_PER_DAY = 1000


def run(seed: int, days: int, shocks: list[FundShock]) -> tuple[ExchangeSimulation, PortfolioFund]:
    sim, fund = mini_bist(seed, shocks)
    sim.run(days * STEPS_PER_DAY)
    return sim, fund


def series(sim: ExchangeSimulation, fund: PortfolioFund) -> dict[str, np.ndarray]:
    h = sim.history
    liquid = fund.params.liquid
    c = fund.params.concentrated
    sold = np.array([[r.sold.get(s, 0) for s in (c, *liquid)] for r in fund.records])
    return {
        "index": np.array([s.index for s in h]),
        "corner": np.array([s.assets[c].last_price for s in h]) * TICK,
        "liquid": np.array([np.mean([s.assets[x].last_price for x in liquid]) for s in h]) * TICK,
        "sold_corner": np.cumsum(sold[:, 0]),
        "sold_liquid": np.cumsum(sold[:, 1:].sum(axis=1)),
        "nav": np.array([r.nav_per_unit for r in fund.records]) * TICK,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=int, default=12)
    ap.add_argument("--shock-day", type=int, default=8)
    ap.add_argument("--cap", type=float, default=0.08)
    ap.add_argument("--redeem", type=float, default=0.30)
    ap.add_argument("--out", type=Path, default=Path("results/m4_contagion.png"))
    args = ap.parse_args()
    T = args.shock_day * STEPS_PER_DAY + 50

    calm_sim, calm_fund = run(args.seed, args.days, [])
    shocks = [FundShock(T, cap_fraction=args.cap), FundShock(T, redeem_fraction=args.redeem)]
    shock_sim, shock_fund = run(args.seed, args.days, shocks)
    calm, shock = series(calm_sim, calm_fund), series(shock_sim, shock_fund)

    v = FUNDAMENTAL * TICK
    c = shock_fund.params.concentrated
    liq0, liq_low = shock["liquid"][T - 1], shock["liquid"][T:].min()
    idx0, idx_low = shock["index"][T - 1], shock["index"][T:].min()
    nav0, nav_low = shock["nav"][T - 1], np.nanmin(shock["nav"][T:])
    sold_c = shock["sold_corner"][-1] - shock["sold_corner"][T - 1]
    sold_l = shock["sold_liquid"][-1] - shock["sold_liquid"][T - 1]
    breakers = [(t, f"{th:.0%}") for t, th in shock_sim.exchange.index_halts]
    rows = [
        ("fundamental value (all stocks)", f"{v:.2f} TRY"),
        (
            f"{c} before / low",
            f"{shock['corner'][T - 1]:.2f} / {shock['corner'][T:].min():.2f} TRY",
        ),
        ("liquid stocks (avg) before / low", f"{liq0:.2f} / {liq_low:.2f} TRY"),
        (
            "  change (calm run)",
            f"{liq_low / liq0 - 1:+.1%} ({calm['liquid'][T:].min() / liq0 - 1:+.1%})",
        ),
        ("index before / low", f"{idx0:.1f} / {idx_low:.1f} ({idx_low / idx0 - 1:+.1%})"),
        ("index circuit breakers", str(breakers)),
        ("  in the calm run", str(calm_sim.exchange.index_halts)),
        ("fund sold after shock", f"{c}: {sold_c:,} shares, liquid: {sold_l:,} shares"),
        ("fund NAV per unit", f"{nav0:.3f} -> low {nav_low:.3f} ({nav_low / nav0 - 1:+.0%})"),
    ]
    print(f"seed {args.seed}: shock at step {T} (day {args.shock_day}), cap {args.cap:.0%}, "
          f"redeem {args.redeem:.0%}")  # fmt: skip
    for label, value in rows:
        print(f"  {label:<34}{value}")

    plot(calm, shock, T, args.out)


def plot(calm: dict[str, np.ndarray], shock: dict[str, np.ndarray], T: int, out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed, skipping figure (pip install -e ".[plot]")')
        return

    blue, orange, gray, ink, muted = "#2a78d6", "#eb6834", "#a3a3a3", "#1f1f1f", "#8a8a8a"
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": muted,
            "xtick.color": muted,
            "ytick.color": muted,
            "axes.grid": True,
            "grid.color": "#e6e6e6",
            "grid.linewidth": 0.6,
            "font.size": 9,
        }
    )
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    days = np.arange(len(shock["index"])) / STEPS_PER_DAY
    shock_day = T / STEPS_PER_DAY

    def pair(a: plt.Axes, key: str) -> None:
        a.plot(days, calm[key], color=gray, lw=1, label="no shock")
        a.plot(days, shock[key], color=blue, lw=1, label="cap + redemptions")
        a.axvline(shock_day, color=muted, lw=0.8, ls=":")

    pair(ax[0, 0], "corner")
    ax[0, 0].set(title="Cornered stock (Ana Pazar)", ylabel="TRY")
    ax[0, 0].legend(frameon=False, loc="upper left")

    pair(ax[0, 1], "liquid")
    ax[0, 1].set(title="Liquid large caps, average (Yildiz Pazar)", ylabel="TRY")

    pair(ax[1, 0], "index")
    prev = shock["index"][(T // STEPS_PER_DAY) * STEPS_PER_DAY - 1]  # previous close
    for th, dy in ((0.05, 3), (0.07, -10)):
        level = prev * (1 - th)
        ax[1, 0].axhline(level, color=ink, lw=0.7, ls="--")
        ax[1, 0].annotate(f"-{th:.0%} breaker", (days[-1], level), xytext=(-60, dy),
                          textcoords="offset points", color=muted)  # fmt: skip
    ax[1, 0].set(title="Index (cap weighted)", xlabel="trading day", ylabel="level")

    ax[1, 1].plot(days, shock["sold_corner"], color=orange, lw=1.2, label="cornered stock")
    ax[1, 1].plot(days, shock["sold_liquid"], color=blue, lw=1.2, label="liquid stocks")
    ax[1, 1].axvline(shock_day, color=muted, lw=0.8, ls=":")
    ax[1, 1].set(title="Shares sold by the fund (cumulative, shocked run)",
                 xlabel="trading day", ylabel="shares")  # fmt: skip
    ax[1, 1].legend(frameon=False, loc="upper left")

    fig.suptitle("ContagionSim M4: a cornered stock's fire sale spreads to the index", color=ink)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"figure saved to {out}")


if __name__ == "__main__":
    main()
