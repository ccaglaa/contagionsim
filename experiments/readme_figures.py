"""Three simple charts for the README, drawn from the experiment results

Reads the CSVs written by q1_phase_diagram.py, q2_liquidation.py and q3_policy.py, so
run those first. The Q4 numbers are copied from the output of q4_early_warning.py
(false alarms at 80% recall on the test runs)

Usage:
    python experiments/readme_figures.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

RESULTS = Path("results")
BLUE = "#2a78d6"
GREY = "#b4b2a9"
INK = "#0b0b0b"
INK_2 = "#52514e"

# Q4, from q4_early_warning.py: false alarms at the threshold that catches 80% of crashes
Q4_FALSE_ALARMS = [
    ("Anyone: simple rule\n(market already down a lot)", 44.0),
    ("Anyone: machine learning\non market data", 4.3),
    ("Regulator: machine learning\n+ the fund's withdrawals", 1.0),
]


def rows(name: str) -> list[dict[str, str]]:
    with open(RESULTS / name, newline="") as f:
        return list(csv.DictReader(f))


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def style(ax: Axes, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=INK, pad=26)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10.5, color=INK_2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=INK_2, length=0, labelsize=10.5)


def q1_tipping_point() -> None:
    halts: dict[float, list[bool]] = defaultdict(list)
    for r in rows("q1_grid.csv"):
        halts[float(r["redemption"])].append(r["index_breaker_5"] == "True")
    levels = sorted(halts)
    share = [100 * mean([float(h) for h in halts[x]]) for x in levels]

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=200)
    labels = [f"{x:.0%}" for x in levels]
    bars = ax.bar(labels, share, color=BLUE, width=0.6)
    for bar, s in zip(bars, share, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            s + 2,
            f"{s:.0f}%",
            ha="center",
            fontsize=10.5,
            color=INK,
        )
    ax.set_ylim(0, 112)
    ax.set_yticks([])
    ax.set_xlabel("Share of the fund's money withdrawn at once", color=INK_2, fontsize=10.5)
    style(
        ax,
        "A run must be big to break the market",
        "Share of simulations that end in a market-wide trading halt",
    )
    fig.tight_layout()
    fig.savefig(RESULTS / "readme_q1.png", facecolor="white")
    plt.close(fig)


def what_limits_the_damage() -> None:
    def liquid_fall(name: str, keep: dict[str, str]) -> float:
        # mean fall of the healthy (liquid) stocks over the seeds of one experiment cell
        xs = [
            -100 * float(r["liquid_drawdown"])
            for r in rows(name)
            if all(r[k] == v for k, v in keep.items())
        ]
        return mean(xs)

    def policy(p: str) -> float:
        return liquid_fall("q3_policy.csv", {"policy": p})

    calm = liquid_fall("q1_grid.csv", {"redemption": "0.0"})
    half_day = liquid_fall(
        "q2_liquidation.csv", {"schedule": "TWAP 0.5d", "scenario": "fire sale only"}
    )
    cases = [
        ("No market-wide halt", policy("no index breaker")),
        ("Fund sells everything at once\n(today's rules)", policy("baseline")),
        ("Fund gets 3 days to cut\nits position", policy("cap phase-in 3d")),
        ("Withdrawals limited to\n20% of the fund per day", policy("gate 20%/day")),
        ("Fund sells over half a day", half_day),
        ("Withdrawals limited to\n10% of the fund per day", policy("gate 10%/day")),
    ]
    labels = [c[0] for c in cases][::-1]
    values = [c[1] for c in cases][::-1]

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    colors = [BLUE if v <= calm + 0.5 else GREY for v in values]
    bars = ax.barh(labels, values, color=colors, height=0.6)
    for bar, v in zip(bars, values, strict=True):
        ax.text(
            max(v, calm) + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.1f}%",
            va="center",
            fontsize=10.5,
            color=INK,
        )
    ax.axvline(calm, color=INK_2, linestyle="--", linewidth=1)
    ax.text(calm + 0.1, -0.75, "normal day", fontsize=9.5, color=INK_2)
    ax.set_ylim(-0.9, len(values) - 0.5)
    ax.set_xlim(0, max(values) * 1.15)
    ax.set_xticks([])
    ax.set_xlabel(
        "How far the healthy big stocks fall (blue: no worse than a normal day)",
        color=INK_2,
        fontsize=10.5,
    )
    style(
        ax,
        "Slowing the selling down protects the market",
        "Same crisis, one thing changed at a time",
    )
    fig.tight_layout()
    fig.savefig(RESULTS / "readme_rules.png", facecolor="white")
    plt.close(fig)


def q4_false_alarms() -> None:
    labels = [x[0] for x in Q4_FALSE_ALARMS][::-1]
    values = [x[1] for x in Q4_FALSE_ALARMS][::-1]
    fig, ax = plt.subplots(figsize=(8, 3.3), dpi=200)
    bars = ax.barh(labels, values, color=[BLUE, GREY, GREY], height=0.6)
    for bar, v in zip(bars, values, strict=True):
        ax.text(
            v + 0.6,
            bar.get_y() + bar.get_height() / 2,
            f"{v:g}%",
            va="center",
            fontsize=10.5,
            color=INK,
        )
    ax.set_xlim(0, 50)
    ax.set_xticks([])
    ax.set_xlabel("False alarms when catching 8 crashes out of 10", color=INK_2, fontsize=10.5)
    style(
        ax,
        "The regulator's data gives the best warning",
        "Alarm up to 50 minutes before a market-wide halt",
    )
    fig.tight_layout()
    fig.savefig(RESULTS / "readme_q4.png", facecolor="white")
    plt.close(fig)


def main() -> None:
    q1_tipping_point()
    what_limits_the_damage()
    q4_false_alarms()
    print("saved results/readme_q1.png, readme_rules.png, readme_q4.png")


if __name__ == "__main__":
    main()
