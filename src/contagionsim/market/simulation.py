"""Discrete-time simulation loop

Each step:
    1. agents act in a random order (fresh shuffle every step, so no agent
       gets a permanent queue advantage from being first in the list)
    2. a Snapshot of the market is recorded

One numpy Generator, created from the seed, drives all randomness:
same seed -> identical run, which is what makes experiments reproducible
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from contagionsim.market.market import Market
from contagionsim.orderbook import Trade


class Agent(Protocol):
    name: str

    def act(self, t: int, market: Market, rng: np.random.Generator) -> None: ...


@dataclass(frozen=True, slots=True)
class Snapshot:
    t: int
    best_bid: int | None
    best_ask: int | None
    mid: float | None
    last_price: int
    volume: int  # shares traded during this step
    n_trades: int
    imbalance: float | None  # top level


def snapshot_of(market: Market, t: int, new_trades: list[Trade]) -> Snapshot:
    """Market state after step t. During an auction the book is crossed: no quotes reported"""
    book = market.book
    quoting = not book.collecting
    return Snapshot(
        t=t,
        best_bid=book.best_bid() if quoting else None,
        best_ask=book.best_ask() if quoting else None,
        mid=book.mid() if quoting else None,
        last_price=market.last_price,
        volume=sum(tr.quantity for tr in new_trades),
        n_trades=len(new_trades),
        imbalance=book.imbalance(1) if quoting else None,
    )


@dataclass
class Simulation:
    market: Market
    agents: list[Agent]
    seed: int
    t: int = 0
    history: list[Snapshot] = field(default_factory=list)
    positions: dict[str, list[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        names = [a.name for a in self.agents]
        if len(set(names)) != len(names):
            raise ValueError("agent names must be unique")
        for name in names:
            if name not in self.market.accounts:
                self.market.register(name)
            self.positions[name] = []

    def step(self) -> Snapshot:
        n_before = len(self.market.book.trades)
        for i in self.rng.permutation(len(self.agents)):
            self.agents[i].act(self.t, self.market, self.rng)
        snap = snapshot_of(self.market, self.t, self.market.book.trades[n_before:])
        self.history.append(snap)
        for name, series in self.positions.items():
            series.append(self.market.accounts[name].position)
        self.t += 1
        return snap

    def run(self, n_steps: int) -> list[Snapshot]:
        for _ in range(n_steps):
            self.step()
        return self.history
