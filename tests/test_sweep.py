"""Sweep tools and contagion metrics"""

from __future__ import annotations

from pathlib import Path

import pytest

from contagionsim.analysis.contagion import contagion_metrics
from contagionsim.scenarios import mini_bist
from contagionsim.sweep import grid, read_csv, run_sweep, write_csv


def test_grid_is_the_cartesian_product_in_order() -> None:
    assert grid(a=[1, 2], b=["x"]) == [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]
    assert len(grid(a=[1, 2, 3], b=[1, 2], c=[0])) == 6


def double(cell: dict[str, int]) -> dict[str, int]:
    return {**cell, "out": 2 * cell["x"]}


def test_run_sweep_keeps_cell_order() -> None:
    rows = run_sweep(double, grid(x=[3, 1, 2]), workers=1, progress=False)
    assert [r["out"] for r in rows] == [6, 2, 4]


def test_csv_round_trip_restores_types(tmp_path: Path) -> None:
    rows = [{"a": 1, "b": 0.5, "c": True, "d": "text"}]
    path = tmp_path / "out" / "rows.csv"
    write_csv(rows, path)
    assert read_csv(path) == rows


def test_write_csv_refuses_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_csv([], tmp_path / "x.csv")


def test_contagion_metrics_on_a_calm_run() -> None:
    sim, fund = mini_bist(seed=0)
    sim.run(1200)
    m = contagion_metrics(sim, fund, t_shock=1000)
    assert m.liquid_drawdown <= 0 and m.index_drawdown <= 0
    assert not m.index_breaker_5 and not m.fund_failed
    assert 0.0 <= m.concentrated_weight <= 1.0
    assert m.concentrated_share_of_sales == 0.0  # no shock: nothing sold
    with pytest.raises(ValueError):
        contagion_metrics(sim, fund, t_shock=5000)
