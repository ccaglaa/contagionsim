"""Q3 (policy): which rules limit the contagion?

The naive fund (dumps immediately) gets the M4 shock on day 8: CORNER cap cut to 8%
of free float + 30% of units redeemed. One rule is changed at a time:

    baseline              today's rules (cap due at once, no gate, CORNER +/-15%, index breaker on)
    cap phase-in 1d / 3d  the regulator lets the fund sell its CORNER excess over 1 or 3 days
    gate 20% / 10%        at most 20% or 10% of NAV can be paid to redeemers per day
    CORNER +/-10% / 20%   CORNER moved to a segment with a narrower or wider daily limit
    CORNER no limits      no daily limit or circuit breaker on CORNER
    no index breaker      the 5% / 7% index circuit breaker switched off

Outcomes (means over seeds): liquid stocks' drawdown (contagion), index drawdown and
5% index breaker, CORNER drawdown, the fund's sale loss vs fair value, and the cost of
gates to investors: redemptions still unpaid one day after the shock

Usage:
    python experiments/q3_policy.py              # 9 policies x 5 seeds = 45 runs
    python experiments/q3_policy.py --seeds 2
    python experiments/q3_policy.py --plot-only
"""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from contagionsim.agents import FundShock
from contagionsim.analysis.contagion import contagion_metrics
from contagionsim.execution import Linear
from contagionsim.market import ALT, NO_LIMITS, YILDIZ
from contagionsim.scenarios import MiniBistConfig, mini_bist
from contagionsim.sweep import grid, read_csv, run_sweep, write_csv

DAY = 1000
SHOCK_T = 8 * DAY + 50
DAYS = 12


def _gate(share: float) -> Callable[[MiniBistConfig], MiniBistConfig]:
    def apply(c: MiniBistConfig) -> MiniBistConfig:
        return dataclasses.replace(c, fund=dataclasses.replace(c.fund, redemption_gate=share))

    return apply


POLICIES: dict[str, Callable[[MiniBistConfig], MiniBistConfig]] = {
    "baseline": lambda c: c,
    "cap phase-in 1d": lambda c: dataclasses.replace(c, cap_schedule=Linear(DAY)),
    "cap phase-in 3d": lambda c: dataclasses.replace(c, cap_schedule=Linear(3 * DAY)),
    "gate 20%/day": _gate(0.20),
    "gate 10%/day": _gate(0.10),
    "CORNER +/-10%": lambda c: dataclasses.replace(c, concentrated_rules=ALT),
    "CORNER +/-20%": lambda c: dataclasses.replace(c, concentrated_rules=YILDIZ),
    "CORNER no limits": lambda c: dataclasses.replace(c, concentrated_rules=NO_LIMITS),
    "no index breaker": lambda c: dataclasses.replace(c, index_rules=None),
}


def run_cell(cell: dict[str, Any]) -> dict[str, Any]:
    config = POLICIES[cell["policy"]](MiniBistConfig())
    shocks = [FundShock(SHOCK_T, cap_fraction=0.08), FundShock(SHOCK_T, redeem_fraction=0.30)]
    sim, fund = mini_bist(cell["seed"], shocks, config)
    sim.run(DAYS * DAY)
    return {**cell, **contagion_metrics(sim, fund, SHOCK_T).as_dict()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--csv", type=Path, default=Path("results/q3_policy.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/q3_policy.png"))
    args = ap.parse_args()

    if args.plot_only:
        rows = read_csv(args.csv)
    else:
        cells = grid(policy=list(POLICIES), seed=list(range(args.seeds)))
        print(f"running {len(cells)} simulations ...")
        rows = run_sweep(run_cell, cells, workers=args.workers)
        write_csv(rows, args.csv)
        print(f"results saved to {args.csv}")

    table = summarize(rows)
    print_table(table)
    plot(table, args.out)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for policy in POLICIES:
        sel = [r for r in rows if r["policy"] == policy]

        def mean(key: str, scale: float = 100.0, rows: list[dict[str, Any]] = sel) -> float:
            return float(np.mean([scale * float(r[key]) for r in rows]))

        def se(key: str, scale: float = 100.0, rows: list[dict[str, Any]] = sel) -> float:
            vals = [scale * float(r[key]) for r in rows]
            return float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0

        out.append(
            {
                "policy": policy,
                "liquid_dd": mean("liquid_drawdown", -100),
                "liquid_se": se("liquid_drawdown", -100),
                "index_dd": mean("index_drawdown", -100),
                "breaker": mean("index_breaker_5"),
                "corner_dd": mean("concentrated_drawdown", -100),
                "fund_cost": mean("sale_loss_vs_value"),
                "owed_1d": mean("owed_after_1_day"),
                "failed": mean("fund_failed"),
            }
        )
    return out


def print_table(table: list[dict[str, Any]]) -> None:
    print("\nmeans over seeds (%). liquid dd = contagion; fund cost = sale loss vs fair value")
    print("(< 0 = sold above value); owed 1d = redemptions still unpaid a day later (% of NAV)")
    print(f"{'policy':<18}{'liquid dd':>10}{'index dd':>10}{'breaker':>9}{'CORNER dd':>11}"
          f"{'fund cost':>11}{'owed 1d':>9}")  # fmt: skip
    for t in table:
        breaker = "n/a" if t["policy"] == "no index breaker" else f"{t['breaker']:.0f}%"
        print(
            f"{t['policy']:<18}{t['liquid_dd']:>10.1f}{t['index_dd']:>10.1f}{breaker:>9}"
            f"{t['corner_dd']:>11.1f}{t['fund_cost']:>+11.2f}{t['owed_1d']:>9.1f}"
        )


def plot(table: list[dict[str, Any]], out: Path) -> None:
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
            "font.size": 9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    names = [t["policy"] for t in table]
    y = np.arange(len(names))[::-1]
    base = table[0]

    panels = (
        ("liquid_dd", "Contagion: liquid stocks' drawdown (%)", "liquid_se"),
        ("owed_1d", "Cost to investors: redemptions unpaid after 1 day (% of NAV)", None),
    )
    for ax, (key, title, err) in zip(axes, panels, strict=True):
        vals = [t[key] for t in table]
        colors = [gray if t["policy"] == "baseline" else blue for t in table]
        xerr = [t[err] for t in table] if err else None
        ax.barh(y, vals, xerr=xerr, color=colors, height=0.6,
                error_kw={"ecolor": muted, "elinewidth": 1})  # fmt: skip
        ax.axvline(base[key], color=ink, lw=0.8, ls="--")
        for yi, v in zip(y, vals, strict=True):
            ax.text(v, yi, f" {v:.1f}", va="center", fontsize=8, color=ink)
        ax.set(title=title)
        ax.grid(axis="x", color="#e6e6e6", lw=0.6)
    axes[0].set_yticks(y, names)
    fig.suptitle("Q3 policy: one rule changed at a time (dashed line = today's rules; "
                 "bars = 1 s.e.)", color=ink)  # fmt: skip
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"figure saved to {out}")


if __name__ == "__main__":
    main()
