"""Q1: when does it break? Concentration x redemption shock phase diagram

Grid:
    concentration : the fund's limit in the cornered stock, as a share of its free float
    redemption    : share of fund units redeemed at once on day 8 (no regulatory cap here,
                    so the two axes are the only things that change)
Each cell is run with the same seeds (common random numbers), so cells differ only in
the two parameters. Outcomes per cell, averaged over seeds:
    - liquid stocks' drawdown after the shock (contagion)
    - probability that the 5% index circuit breaker fires

Usage:
    python experiments/q1_phase_diagram.py              # full grid: 6 x 6 x 3 seeds = 108 runs
    python experiments/q1_phase_diagram.py --quick      # 3 x 3 x 1 seed, to check it works
    python experiments/q1_phase_diagram.py --seeds 10   # more seeds per cell, tighter estimates
    python experiments/q1_phase_diagram.py --plot-only  # redraw from results/q1_grid.csv

Runtime: one run ~10 s; the full grid uses every CPU core (about 2-3 min on 8 cores)
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from contagionsim.agents import FundShock
from contagionsim.analysis.contagion import contagion_metrics
from contagionsim.scenarios import MiniBistConfig, mini_bist
from contagionsim.sweep import Row, grid, read_csv, run_sweep, write_csv

STEPS_PER_DAY = 1000
SHOCK_T = 8 * STEPS_PER_DAY + 50
DAYS = 10

FULL = {
    "concentration": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "redemption": [0.0, 0.10, 0.20, 0.30, 0.45, 0.60],
    "seed": [0, 1, 2],
}
QUICK = {"concentration": [0.05, 0.15, 0.30], "redemption": [0.0, 0.30, 0.60], "seed": [0]}


def run_cell(cell: Row) -> Row:
    """One simulation. Top-level so it can run in a worker process"""
    base = MiniBistConfig()
    fund = dataclasses.replace(base.fund, max_float_share=cell["concentration"])
    config = dataclasses.replace(base, fund=fund)
    shocks = [FundShock(SHOCK_T, redeem_fraction=cell["redemption"])] if cell["redemption"] else []
    sim, f = mini_bist(cell["seed"], shocks, config)
    sim.run(DAYS * STEPS_PER_DAY)
    return {**cell, **contagion_metrics(sim, f, SHOCK_T).as_dict()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None, help="seeds per cell (default 3)")
    ap.add_argument("--csv", type=Path, default=Path("results/q1_grid.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/q1_phase_diagram.png"))
    args = ap.parse_args()

    if args.plot_only:
        rows = read_csv(args.csv)
    else:
        axes = dict(QUICK if args.quick else FULL)
        if args.seeds is not None:
            axes["seed"] = list(range(args.seeds))
        cells = grid(**axes)
        print(f"running {len(cells)} simulations ...")
        rows = run_sweep(run_cell, cells, workers=args.workers)
        write_csv(rows, args.csv)
        print(f"results saved to {args.csv}")

    conc = sorted({r["concentration"] for r in rows})
    red = sorted({r["redemption"] for r in rows})
    liquid = cell_means(rows, conc, red, lambda r: -100 * r["liquid_drawdown"])
    breaker = cell_means(rows, conc, red, lambda r: 100 * float(r["index_breaker_5"]))
    diverted = cell_means(
        rows,
        conc,
        red,
        lambda r: 100 * (r["concentrated_weight"] - r["concentrated_share_of_sales"]),
    )
    diverted[:, [j for j, r in enumerate(red) if r == 0]] = np.nan  # nothing sold: undefined
    summary(conc, red, liquid, breaker, diverted)
    plot(conc, red, liquid, breaker, args.out)


def cell_means(rows: list[Row], conc: list[float], red: list[float], value) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Matrix [concentration, redemption] of the mean of value(row) over seeds"""
    out = np.full((len(conc), len(red)), np.nan)
    for i, c in enumerate(conc):
        for j, r in enumerate(red):
            vals = [
                value(row) for row in rows if row["concentration"] == c and row["redemption"] == r
            ]
            out[i, j] = np.mean(vals)
    return out


def summary(
    conc: list[float],
    red: list[float],
    liquid: np.ndarray,
    breaker: np.ndarray,
    diverted: np.ndarray,
) -> None:
    header = "conc \\ redeem " + "".join(f"{r:>8.0%}" for r in red)
    print("\nliquid stocks: average drawdown after the shock (%)")
    print(header)
    for i, c in enumerate(conc):
        print(f"{c:>13.0%} " + "".join(f"{v:8.1f}" for v in liquid[i]))
    print("\nindex 5% circuit breaker: share of seeds where it fired (%)")
    print(header)
    for i, c in enumerate(conc):
        print(f"{c:>13.0%} " + "".join(f"{v:8.0f}" for v in breaker[i]))
    print("\nselling diverted to liquid stocks: concentrated weight - its share of sales (pp)")
    print(header)
    for i, c in enumerate(conc):
        print(
            f"{c:>13.0%} "
            + "".join("       -" if np.isnan(v) else f"{v:8.1f}" for v in diverted[i])
        )


def plot(
    conc: list[float], red: list[float], liquid: np.ndarray, breaker: np.ndarray, out: Path
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed, skipping figure (pip install -e ".[plot]")')
        return

    ink, muted = "#1f1f1f", "#8a8a8a"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    panels = (
        (liquid, "Liquid stocks: drawdown after the shock (%)", "{:.1f}"),
        (breaker, "Index 5% circuit breaker fired (% of seeds)", "{:.0f}"),
    )
    for ax, (data, title, fmt) in zip(axes, panels, strict=True):
        vmax = max(float(np.nanmax(data)), 1e-9)
        im = ax.imshow(data, origin="lower", cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(red)), [f"{r:.0%}" for r in red])
        ax.set_yticks(range(len(conc)), [f"{c:.0%}" for c in conc])
        ax.set(title=title, xlabel="redemption shock (share of units)",
               ylabel="fund concentration (share of free float)")  # fmt: skip
        for i in range(len(conc)):
            for j in range(len(red)):
                v = data[i, j]
                color = "white" if v > 0.6 * vmax else ink
                ax.text(j, i, fmt.format(v), ha="center", va="center", color=color, fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(
        "Q1: when does a concentrated fund's redemption run spread to the index?", color=ink
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"figure saved to {out}")


if __name__ == "__main__":
    main()
