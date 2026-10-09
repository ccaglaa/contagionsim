"""M3: build-up and forced unwind of a concentrated fund

Two runs with the same seed, identical until the shock:
    calm    : the fund keeps growing, nothing happens
    shocked : at --shock-t a regulator caps holdings at 8% of free float and
              30% of units are redeemed at once; the fund sells naively

Usage:
    python experiments/m3_fund_unwind.py
    python experiments/m3_fund_unwind.py --seed 3 --cap 0.05 --redeem 0.5

Prints a summary and saves results/m3_fund_unwind.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from contagionsim.agents import (
    Fund,
    FundParams,
    FundShock,
    MarketMaker,
    NoiseTrader,
    ValueTrader,
    ValueTraderParams,
)
from contagionsim.market import Agent, Market, Simulation

TICK = 0.01  # TRY per tick
FUNDAMENTAL = 1000  # ticks = 10.00 TRY
FREE_FLOAT = 100_000


def build(seed: int, shocks: list[FundShock]) -> tuple[Simulation, Fund]:
    agents: list[Agent] = [NoiseTrader(f"noise{i}") for i in range(20)]
    agents.append(MarketMaker("mm"))
    agents += [
        ValueTrader(f"value{i}", ValueTraderParams(fundamental=FUNDAMENTAL)) for i in range(5)
    ]
    fund = Fund("fund", FundParams(free_float=FREE_FLOAT), shocks)
    agents.append(fund)
    return Simulation(Market(initial_price=FUNDAMENTAL), agents, seed=seed), fund


def series(sim: Simulation, fund: Fund) -> dict[str, np.ndarray]:
    hist = sim.history
    return {
        "price": np.array([s.mid if s.mid is not None else s.last_price for s in hist]) * TICK,
        "spread": np.array(
            [s.best_ask - s.best_bid if s.best_ask and s.best_bid else np.nan for s in hist]
        ),
        "holding": np.array([r.shares for r in fund.records]) / FREE_FLOAT * 100,
        "nav": np.array([r.nav_per_unit for r in fund.records]) * TICK,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=14_000)
    ap.add_argument("--shock-t", type=int, default=8_050)
    ap.add_argument("--cap", type=float, default=0.08, help="new cap, share of free float")
    ap.add_argument("--redeem", type=float, default=0.30, help="share of units redeemed")
    ap.add_argument("--out", type=Path, default=Path("results/m3_fund_unwind.png"))
    args = ap.parse_args()
    T = args.shock_t

    calm_sim, calm_fund = build(args.seed, [])
    calm_sim.run(args.steps)
    shocks = [FundShock(T, cap_fraction=args.cap), FundShock(T, redeem_fraction=args.redeem)]
    shock_sim, shock_fund = build(args.seed, shocks)
    shock_sim.run(args.steps)

    calm, shock = series(calm_sim, calm_fund), series(shock_sim, shock_fund)
    v = FUNDAMENTAL * TICK
    low = shock["price"][T:].min()
    value_pos = [shock_sim.market.accounts[f"value{i}"].position for i in range(5)]
    sold = sum(r.shares_sold for r in shock_fund.records[T:])

    print(f"seed {args.seed}: shock at step {T} (cap {args.cap:.0%}, redeem {args.redeem:.0%})")
    print(f"  fundamental value        {v:.2f} TRY")
    print(
        f"  price before shock       {shock['price'][T - 1]:.2f} TRY "
        f"({shock['price'][T - 1] / v - 1:+.0%} vs value)"
    )
    print(f"  fund holding before      {shock['holding'][T - 1]:.1f}% of free float")
    print(f"  lowest price after       {low:.2f} TRY ({low / v - 1:+.0%} vs value)")
    print(
        f"  price at end             {shock['price'][-1]:.2f} TRY "
        f"(calm run: {calm['price'][-1]:.2f})"
    )
    print(f"  shares sold by the fund  {sold:,}")
    nav_t, nav_low = shock["nav"][T - 1], np.nanmin(shock["nav"][T:])
    print(
        f"  NAV per unit             {nav_t:.3f} -> low {nav_low:.3f} TRY "
        f"({nav_low / nav_t - 1:+.0%})"
    )
    print(f"  value traders long       {sum(value_pos):,} of {5 * 5000:,} capacity")
    print(f"  fund failed              {shock_fund.failed_at is not None}")

    plot(calm, shock, T, args.cap * 100, v, args.out)


def plot(
    calm: dict[str, np.ndarray],
    shock: dict[str, np.ndarray],
    T: int,
    cap_pct: float,
    value: float,
    out: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed, skipping figure (pip install -e ".[plot]")')
        return

    blue, gray, ink, muted = "#2a78d6", "#a3a3a3", "#1f1f1f", "#8a8a8a"
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
    t = np.arange(len(shock["price"]))

    def pair(a: plt.Axes, key: str, smooth: int = 1) -> None:
        for data, color, label in ((calm, gray, "no shock"), (shock, blue, "cap + redemptions")):
            y, x = data[key], t
            if smooth > 1:
                # trailing moving average, only where the full window exists
                y = np.convolve(np.nan_to_num(y), np.ones(smooth) / smooth, mode="valid")
                x = t[smooth - 1 :]
            a.plot(x, y, color=color, lw=1, label=label)
        a.axvline(T, color=muted, lw=0.8, ls=":")

    pair(ax[0, 0], "price")
    ax[0, 0].axhline(value, color=ink, lw=0.8, ls="--")
    ax[0, 0].annotate(
        "fundamental value", (0, value), xytext=(4, -12), textcoords="offset points", color=muted
    )
    ax[0, 0].set(title="Price", ylabel="TRY")
    ax[0, 0].legend(frameon=False, loc="upper left")

    pair(ax[0, 1], "holding")
    ax[0, 1].hlines(cap_pct, T, t[-1], color=ink, lw=0.8, ls="--")
    ax[0, 1].annotate(
        "new cap", (T, cap_pct), xytext=(4, -12), textcoords="offset points", color=muted
    )
    ax[0, 1].set(title="Fund holding", ylabel="% of free float")

    pair(ax[1, 0], "nav")
    ax[1, 0].set(title="Fund NAV per unit", xlabel="step", ylabel="TRY")

    pair(ax[1, 1], "spread", smooth=100)
    ax[1, 1].set(title="Bid-ask spread (100-step average)", xlabel="step", ylabel="ticks")

    fig.suptitle("ContagionSim M3: build-up and forced unwind of a concentrated fund", color=ink)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"figure saved to {out}")


if __name__ == "__main__":
    main()
