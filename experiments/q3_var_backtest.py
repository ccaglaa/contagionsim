"""Q3: what do standard risk models miss? VaR backtest through the fire sale

For each seed, two runs of the mini BIST (identical until day 8):
    calm  : nothing happens
    shock : the M4 shock (CORNER cap cut to 8% of float + 30% of fund units redeemed)
Prices are cut into bars of 50 steps (20 bars per trading day). For three series
(the index, the liquid stocks' average, CORNER) and three models:
    historical simulation, normal (rolling std), GARCH(1,1)
the models are estimated on days 1-7 (140 bars; day 0 is start-up and is skipped)
and backtested on days 8-9 (40 bars).
Exceptions are pooled over seeds; Kupiec (count) and Christoffersen (clustering) tests.

The calm runs are the control: a model that fails there is simply a bad model.
A model that passes in calm and fails in the shock is blind to this kind of event.

Usage:
    python experiments/q3_var_backtest.py              # 10 seeds x 2 scenarios
    python experiments/q3_var_backtest.py --seeds 3    # faster check
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from contagionsim.agents import FundShock
from contagionsim.analysis.var import backtest, fit_garch11, historical_var, hits, normal_var
from contagionsim.scenarios import mini_bist
from contagionsim.sweep import grid, run_sweep, write_csv

STEPS_PER_DAY = 1000
BAR = 50  # steps per bar
DAYS = 10
SHOCK_T = 8 * STEPS_PER_DAY + 50
TEST_START = 8 * STEPS_PER_DAY // BAR  # first bar of day 8 = 160
BURN_IN = STEPS_PER_DAY // BAR  # day 0 is start-up (agents filling the book): not used
WINDOW = TEST_START - BURN_IN  # estimation window: days 1-7 = 140 bars
LEVELS = (0.99, 0.95)
MODELS = ("historical", "normal", "garch")
SERIES = ("index", "liquid", "corner")


def bar_returns(prices: np.ndarray) -> np.ndarray:
    """Log returns between bar closes; bar k ends at step (k + 1) * BAR - 1"""
    closes = prices[BAR - 1 :: BAR]
    return np.diff(np.log(np.r_[prices[0], closes]))


def run_cell(cell: dict[str, Any]) -> dict[str, Any]:
    shocks = []
    if cell["scenario"] == "shock":
        shocks = [FundShock(SHOCK_T, cap_fraction=0.08), FundShock(SHOCK_T, redeem_fraction=0.30)]
    sim, fund = mini_bist(cell["seed"], shocks)
    sim.run(DAYS * STEPS_PER_DAY)
    h = sim.history
    liquid = fund.params.liquid
    prices = {
        "index": np.array([s.index for s in h]),
        "liquid": np.array([np.mean([s.assets[x].last_price for x in liquid]) for s in h]),
        "corner": np.array([s.assets[fund.params.concentrated].last_price for s in h], float),
    }
    out: dict[str, Any] = {**cell}
    for name, p in prices.items():
        r = bar_returns(p)
        test = r[TEST_START:]
        out[f"{name}_returns"] = test
        garch = fit_garch11(r[BURN_IN:TEST_START])
        r_used = r[BURN_IN:]  # GARCH runs forward from the start of its own window
        for level in LEVELS:
            var = {
                "historical": historical_var(r, WINDOW, level)[TEST_START:],
                "normal": normal_var(r, WINDOW, level)[TEST_START:],
                "garch": garch.var(r_used, level)[WINDOW:],
            }
            for model, v in var.items():
                out[f"{name}_{model}_{level}_var"] = v
                out[f"{name}_{model}_{level}_hits"] = hits(test, v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/q3_var_backtest.png"))
    ap.add_argument("--csv", type=Path, default=Path("results/q3_backtest.csv"))
    args = ap.parse_args()

    cells = grid(seed=list(range(args.seeds)), scenario=["calm", "shock"])
    print(f"running {len(cells)} simulations ...")
    runs = run_sweep(run_cell, cells, workers=args.workers)

    rows = []
    for series in SERIES:
        for level in LEVELS:
            for model in MODELS:
                for scenario in ("calm", "shock"):
                    sel = [r for r in runs if r["scenario"] == scenario]
                    key = f"{series}_{model}_{level}"
                    bt = backtest([r[f"{key}_hits"] for r in sel], level)
                    # Severity: worst loss relative to the VaR that was supposed to cover it
                    worst = max(
                        float(np.max(-r[f"{series}_returns"] / r[f"{key}_var"])) for r in sel
                    )
                    rows.append(
                        {
                            "series": series,
                            "level": level,
                            "model": model,
                            "scenario": scenario,
                            "n": bt.n,
                            "exceptions": bt.exceptions,
                            "expected": round(bt.expected, 1),
                            "kupiec_p": round(bt.kupiec_p, 4),
                            "christoffersen_p": round(bt.christoffersen_p, 4),
                            "cc_p": round(bt.cc_p, 4),
                            "rejected": bt.rejected(),
                            "worst_loss_over_var": round(worst, 2),
                        }
                    )
    write_csv(rows, args.csv)
    print_table(rows)
    print(f"\ntable saved to {args.csv}")
    plot(runs, rows, args.out)


def print_table(rows: list[dict[str, Any]]) -> None:
    print("\nVaR backtest, days 8-9, exceptions pooled over seeds")
    print("(rejected = conditional coverage test, 5%; worst = largest loss / VaR on that bar)")
    head = f"{'series':<7}{'level':>6}{'model':>12}  {'calm: exc/exp':>14}{'rej':>5}"
    print(head + f"  {'shock: exc/exp':>15}{'rej':>5}{'worst':>7}")
    for series in SERIES:
        for level in LEVELS:
            for model in MODELS:
                c, s = (
                    next(
                        r
                        for r in rows
                        if (r["series"], r["level"], r["model"], r["scenario"])
                        == (series, level, model, sc)
                    )
                    for sc in ("calm", "shock")
                )
                calm = f"{c['exceptions']:>6} / {c['expected']:<5}{_yes(c['rejected']):>5}"
                shock = f"{s['exceptions']:>6} / {s['expected']:<6}{_yes(s['rejected']):>5}"
                worst = f"{s['worst_loss_over_var']:>7.1f}"
                print(f"{series:<7}{level:>6.0%}{model:>12}  {calm}  {shock}{worst}")


def _yes(flag: bool) -> str:
    return "yes" if flag else "no"


def plot(runs: list[dict[str, Any]], rows: list[dict[str, Any]], out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed, skipping figure (pip install -e ".[plot]")')
        return

    gray, ink, muted = "#a3a3a3", "#1f1f1f", "#8a8a8a"
    colors = {"historical": "#2a78d6", "normal": "#eb6834", "garch": "#1baf7a"}
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
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})

    # Left: seed 0, shocked run, index returns in the test window with each model's VaR
    run = next(r for r in runs if r["seed"] == 0 and r["scenario"] == "shock")
    r = run["index_returns"] * 100
    days = 8 + np.arange(len(r)) * BAR / STEPS_PER_DAY
    width = BAR / STEPS_PER_DAY * 0.8
    ax[0].bar(days, r, width=width, color=gray, label="index return per bar")
    for model, color in colors.items():
        ax[0].step(days, -run[f"index_{model}_0.99_var"] * 100, where="mid", color=color, lw=1.4,
                   label=f"-VaR 99% ({model})")  # fmt: skip
    hit_any = run["index_garch_0.99_hits"] | run["index_historical_0.99_hits"]
    ax[0].scatter(days[hit_any], r[hit_any], color=ink, s=16, zorder=3, label="exception")
    ax[0].axhline(0, color=muted, lw=0.6)
    ax[0].set(title="Index, seed 0: returns vs 99% VaR through the shock",
              xlabel="trading day", ylabel="% per bar (50 steps)")  # fmt: skip
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")

    # Right: pooled 99% exceptions on the index, calm vs shock, per model
    x = np.arange(len(MODELS))
    for k, (scenario, color) in enumerate((("calm", gray), ("shock", "#2a78d6"))):
        vals = [
            next(
                row["exceptions"]
                for row in rows
                if (row["series"], row["level"], row["model"], row["scenario"])
                == ("index", 0.99, m, scenario)
            )
            for m in MODELS
        ]
        bars = ax[1].bar(x + (k - 0.5) * 0.38, vals, width=0.36, color=color, label=scenario)
        ax[1].bar_label(bars, padding=2, fontsize=8, color=ink)
    expected = next(row["expected"] for row in rows if row["level"] == 0.99)
    ax[1].axhline(expected, color=ink, lw=0.8, ls="--")
    ax[1].annotate(f"expected at 99%: {expected:g}", (-0.55, expected), xytext=(0, 4),
                   textcoords="offset points", color=muted, fontsize=8)  # fmt: skip
    ax[1].set_xticks(x, MODELS)
    ax[1].set(title="Index: 99% VaR exceptions, all seeds pooled", ylabel="exceptions")
    ax[1].legend(frameon=False, loc="upper left")

    fig.suptitle("Q3: risk models calibrated on calm days are blind to a fire sale", color=ink)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"figure saved to {out}")


if __name__ == "__main__":
    main()
