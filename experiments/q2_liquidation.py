"""Q2: how should a forced seller liquidate? Fund cost vs market cost of each schedule

Shock (day 8): CORNER cap cut to 8% of float + 30% of fund units redeemed (as in M4).
The fund follows one liquidation schedule (execution/schedules.py):
    immediate                  dump everything now (the M4 fund)
    TWAP 0.5 / 1 / 3 days      equal amounts per step over the horizon
    Almgren-Chriss 3 days      optimal trajectory, urgency k = 3 (patient) or 10 (urgent)
Two scenarios:
    fire sale only   : nothing changes in fundamentals, the price drop is pure impact
    bad news         : at the shock the market learns CORNER is worth 20% less

Two costs, averaged over seeds:
    fund   : sale loss vs fair value (% of pre-shock NAV). Fair value, not the market
             mark: a slow seller's unsold shares are marked at inflated prices, which
             would make its NAV look better than its trading really was
    market : drawdown of the liquid stocks (the ones with nothing wrong with them),
             and how often the 5% index circuit breaker fired

Usage:
    python experiments/q2_liquidation.py             # 6 schedules x 2 scenarios x 5 seeds
    python experiments/q2_liquidation.py --seeds 2   # quicker
    python experiments/q2_liquidation.py --plot-only
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from contagionsim.agents import FundShock
from contagionsim.analysis.contagion import contagion_metrics
from contagionsim.execution import AlmgrenChriss, Immediate, Linear, Schedule
from contagionsim.scenarios import FUNDAMENTAL, MiniBistConfig, News, mini_bist
from contagionsim.sweep import grid, read_csv, run_sweep, write_csv

DAY = 1000
SHOCK_T = 8 * DAY + 50
DAYS = 13  # 5 days after the shock: every schedule has finished
BAD_NEWS_FACTOR = 0.8

SCHEDULES: dict[str, Schedule] = {
    "immediate": Immediate(),
    "TWAP 0.5d": Linear(DAY // 2),
    "TWAP 1d": Linear(DAY),
    "TWAP 3d": Linear(3 * DAY),
    "AC 3d k=3": AlmgrenChriss(3 * DAY, 3.0),
    "AC 3d k=10": AlmgrenChriss(3 * DAY, 10.0),
}
SCENARIOS = ("fire sale only", "bad news")


def run_cell(cell: dict[str, Any]) -> dict[str, Any]:
    config = dataclasses.replace(MiniBistConfig(), schedule=SCHEDULES[cell["schedule"]])
    shocks = [FundShock(SHOCK_T, cap_fraction=0.08), FundShock(SHOCK_T, redeem_fraction=0.30)]
    bad = cell["scenario"] == "bad news"
    news = [News(SHOCK_T, config.concentrated, BAD_NEWS_FACTOR)] if bad else None
    sim, fund = mini_bist(cell["seed"], shocks, config, news)
    sim.run(DAYS * DAY)
    fair = {config.concentrated: FUNDAMENTAL * BAD_NEWS_FACTOR} if bad else None
    return {**cell, **contagion_metrics(sim, fund, SHOCK_T, fair).as_dict()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--csv", type=Path, default=Path("results/q2_liquidation.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/q2_liquidation.png"))
    args = ap.parse_args()

    if args.plot_only:
        rows = read_csv(args.csv)
    else:
        cells = grid(
            scenario=list(SCENARIOS), schedule=list(SCHEDULES), seed=list(range(args.seeds))
        )
        print(f"running {len(cells)} simulations ...")
        rows = run_sweep(run_cell, cells, workers=args.workers)
        write_csv(rows, args.csv)
        print(f"results saved to {args.csv}")

    table = summarize(rows)
    print_table(table)
    plot(table, args.out)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for scenario in SCENARIOS:
        for schedule in SCHEDULES:
            sel = [r for r in rows if r["scenario"] == scenario and r["schedule"] == schedule]
            fund = np.array([100 * r["sale_loss_vs_value"] for r in sel])
            market = np.array([-100 * r["liquid_drawdown"] for r in sel])
            out.append(
                {
                    "scenario": scenario,
                    "schedule": schedule,
                    "fund_cost": fund.mean(),
                    "fund_se": fund.std(ddof=1) / np.sqrt(len(fund)) if len(fund) > 1 else 0.0,
                    "market_cost": market.mean(),
                    "market_se": market.std(ddof=1) / np.sqrt(len(market))
                    if len(market) > 1
                    else 0.0,
                    "breaker": 100 * np.mean([r["index_breaker_5"] for r in sel]),
                    "corner_dd": -100 * np.mean([r["concentrated_drawdown"] for r in sel]),
                    "nav_dd": -100 * np.mean([r["nav_drawdown"] for r in sel]),
                }
            )
    return out


def print_table(table: list[dict[str, Any]]) -> None:
    print("\nmeans over seeds. fund cost = sale loss vs fair value (% of NAV, negative = sold")
    print("above fair value); market cost = liquid stocks' drawdown (%)")
    for scenario in SCENARIOS:
        print(f"\n{scenario}")
        print(f"{'schedule':<12}{'fund cost':>11}{'market cost':>13}{'breaker 5%':>12}"
              f"{'CORNER dd':>11}{'NAV dd':>9}")  # fmt: skip
        for t in (t for t in table if t["scenario"] == scenario):
            print(
                f"{t['schedule']:<12}{t['fund_cost']:>+11.2f}{t['market_cost']:>13.2f}"
                f"{t['breaker']:>11.0f}%{t['corner_dd']:>11.1f}{t['nav_dd']:>9.1f}"
            )


def plot(table: list[dict[str, Any]], out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed, skipping figure (pip install -e ".[plot]")')
        return

    ink, muted = "#1f1f1f", "#8a8a8a"
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
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
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, scenario in zip(axes, SCENARIOS, strict=True):
        for color, schedule in zip(colors, SCHEDULES, strict=True):
            t = next(x for x in table if x["scenario"] == scenario and x["schedule"] == schedule)
            ax.errorbar(t["fund_cost"], t["market_cost"], xerr=t["fund_se"], yerr=t["market_se"],
                        fmt="o", ms=8, color=color, ecolor=color, elinewidth=1, capsize=2,
                        label=schedule)  # fmt: skip
            if schedule == "immediate":  # the cluster is identified by the legend
                ax.annotate("dump immediately", (t["fund_cost"], t["market_cost"]),
                            xytext=(-30, 10), textcoords="offset points", fontsize=8,
                            color=ink)  # fmt: skip
        spread = [x for x in table if x["scenario"] == scenario and x["schedule"] != "immediate"]
        ax.annotate("spread over time", (np.mean([x["fund_cost"] for x in spread]),
                    max(x["market_cost"] for x in spread)), xytext=(-40, 18),
                    textcoords="offset points", fontsize=8, color=muted)  # fmt: skip
        ax.set(title=scenario, xlabel="fund cost: sale loss vs fair value (% of NAV, < 0 = gain)")
        ax.axvline(0, color=muted, lw=0.6)
    axes[0].set_ylabel("market cost: liquid stocks' drawdown (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Q2: liquidation schedules, cost to the fund vs cost to the market "
                 "(lower left is better; bars = 1 s.e.)", color=ink)  # fmt: skip
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"figure saved to {out}")


if __name__ == "__main__":
    main()
