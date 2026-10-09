"""M2 baseline: noise traders + one market maker, no fund

Usage:
    python experiments/m2_baseline.py                 # default seed 0, 20000 steps
    python experiments/m2_baseline.py --seed 3 --steps 50000

Prints market statistics and stylized facts, saves results/m2_baseline.png
This run is the "calm market" every later experiment is compared against
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from contagionsim.agents import MarketMaker, NoiseTrader
from contagionsim.analysis import stylized_facts
from contagionsim.market import Agent, Market, Simulation

TICK = 0.01  # TRY per tick, so 1000 ticks = 10.00 TRY


def build(seed: int, n_noise: int) -> Simulation:
    agents: list[Agent] = [NoiseTrader(f"noise{i}") for i in range(n_noise)]
    agents.append(MarketMaker("mm"))
    return Simulation(Market(initial_price=1000), agents, seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--noise", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("results/m2_baseline.png"))
    args = ap.parse_args()

    sim = build(args.seed, args.noise)
    hist = sim.run(args.steps)
    market = sim.market

    t = np.array([s.t for s in hist])
    ref = np.array([s.mid if s.mid is not None else s.last_price for s in hist], dtype=float)
    spread = np.array(
        [s.best_ask - s.best_bid if s.best_ask and s.best_bid else np.nan for s in hist]
    )
    inv = np.array(sim.positions["mm"])
    total_vol = sum(s.volume for s in hist)
    mm_vol = sum(tr.quantity for tr in market.book.trades if "mm" in (tr.buyer, tr.seller))
    mm_pnl = market.accounts["mm"].value(market.last_price) * TICK

    print(f"seed {args.seed}, {args.steps} steps, {args.noise} noise traders + 1 market maker")
    print(f"  price           {ref[0] * TICK:.2f} -> {ref[-1] * TICK:.2f} TRY")
    print(f"  price range     {ref.min() * TICK:.2f} .. {ref.max() * TICK:.2f} TRY")
    print(f"  mean spread     {np.nanmean(spread):.2f} ticks")
    print(f"  two-sided book  {np.mean(~np.isnan(spread)):.1%} of steps")
    print(f"  volume          {total_vol / args.steps:.0f} shares/step")
    print(f"  MM share        {mm_vol / max(total_vol, 1):.1%} of volume")
    print(f"  MM inventory    {inv.min()} .. {inv.max()} shares")
    print(f"  MM PnL          {mm_pnl:,.0f} TRY")
    print()
    for h in (1, 20):
        print(stylized_facts(ref, horizon=h).summary())
        print()

    plot(t, ref * TICK, spread, inv, ref, args.out)


def plot(
    t: np.ndarray,
    price: np.ndarray,
    spread: np.ndarray,
    inv: np.ndarray,
    ref: np.ndarray,
    out: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed, skipping figure (pip install -e ".[plot]")')
        return

    blue, ink, muted = "#2a78d6", "#1f1f1f", "#8a8a8a"
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": muted,
            "axes.labelcolor": ink,
            "xtick.color": muted,
            "ytick.color": muted,
            "axes.grid": True,
            "grid.color": "#e6e6e6",
            "grid.linewidth": 0.6,
            "font.size": 9,
        }
    )
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))

    ax[0, 0].plot(t, price, color=blue, lw=1)
    ax[0, 0].set(title="Mid price", xlabel="step", ylabel="TRY")

    window = 50
    smooth = np.convolve(np.nan_to_num(spread, nan=0.0), np.ones(window) / window, mode="valid")
    ax[0, 1].plot(t[window - 1 :], smooth, color=blue, lw=1)
    ax[0, 1].set(title=f"Bid-ask spread ({window}-step average)", xlabel="step", ylabel="ticks")

    ax[1, 0].plot(t, inv, color=blue, lw=1)
    ax[1, 0].axhline(0, color=muted, lw=0.8)
    ax[1, 0].set(title="Market maker inventory", xlabel="step", ylabel="shares")

    r = np.diff(np.log(ref[::20]))
    z = (r - r.mean()) / r.std()
    bins = np.linspace(-6, 6, 49)
    ax[1, 1].hist(z, bins=bins, density=True, color=blue, alpha=0.8, label="simulated")
    grid = np.linspace(-6, 6, 400)
    ax[1, 1].plot(
        grid, np.exp(-(grid**2) / 2) / np.sqrt(2 * np.pi), color=ink, lw=1.2, label="normal"
    )
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_ylim(1e-4, 1)
    ax[1, 1].set(title="20-step returns, standardized (log scale)", xlabel="z", ylabel="density")
    ax[1, 1].legend(frameon=False)

    fig.suptitle("ContagionSim M2 baseline: noise traders + market maker", color=ink)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"figure saved to {out}")


if __name__ == "__main__":
    main()
