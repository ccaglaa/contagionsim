"""Parallel parameter sweeps

Each cell of a grid is one independent simulation, so cells run in separate
processes (one per CPU core). A cell is a plain dict of parameters; the worker
function must be a top-level function so it can be sent to another process

Reproducibility: every cell carries its own seed, so the result of a cell does not
depend on which process ran it or in which order. Using the same seeds in every
cell ("common random numbers") makes cells directly comparable: two cells with the
same seed share the same noise and differ only in the parameters being swept
"""

from __future__ import annotations

import csv
import itertools
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

Row = dict[str, Any]


def grid(**axes: Sequence[Any]) -> list[Row]:
    """Cartesian product of the axes: grid(a=[1, 2], b=[3]) -> [{a:1, b:3}, {a:2, b:3}]"""
    names = list(axes)
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*axes.values())]


def run_sweep(
    worker: Callable[[Row], Row],
    cells: Iterable[Row],
    workers: int | None = None,
    progress: bool = True,
) -> list[Row]:
    """Run worker(cell) for every cell, in parallel. Results keep the order of `cells`"""
    cells = list(cells)
    n = workers or os.cpu_count() or 1
    results: list[Row] = []
    if n == 1:
        for i, cell in enumerate(cells, 1):
            results.append(worker(cell))
            _report(progress, i, len(cells))
        return results
    with ProcessPoolExecutor(max_workers=n) as pool:
        for i, row in enumerate(pool.map(worker, cells), 1):
            results.append(row)
            _report(progress, i, len(cells))
    return results


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("no rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[Row]:
    """Read rows back, converting numbers and booleans"""
    with path.open(newline="") as f:
        return [{k: _parse(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _parse(value: str) -> Any:
    if value in ("True", "False"):
        return value == "True"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def _report(progress: bool, done: int, total: int) -> None:
    if progress:
        print(f"\r  {done}/{total} runs done", end="" if done < total else "\n", flush=True)
