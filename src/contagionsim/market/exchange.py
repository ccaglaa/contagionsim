"""Multi-asset exchange: several Markets sharing one Ledger, a mini index, index circuit breaker

Each step (ExchangeSimulation):
    1. exchange.begin_step(t): new trading day? -> new base prices and limits;
       running auctions whose time is up -> uncross
    2. agents act (in a fresh random order)
    3. exchange.end_step(t): index circuit breaker check
    4. snapshot

The index is capitalization weighted, like BIST 100, normalized to 100 at the start:
    index = 100 * sum(shares_i * price_i) / sum(shares_i * initial_price_i)

Agents that trade one asset (noise traders, market makers, value traders) are
wrapped in OnAsset(symbol, agent); multi-asset agents (the fund) receive the Exchange
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from contagionsim.market.market import Ledger, Market
from contagionsim.market.rules import Clock, IndexBreakerRules, SegmentRules
from contagionsim.market.simulation import Snapshot, snapshot_of


@dataclass(frozen=True, slots=True)
class Listing:
    symbol: str
    initial_price: int  # ticks
    shares_outstanding: int  # index weight
    rules: SegmentRules


class Exchange:
    def __init__(
        self,
        listings: list[Listing],
        index_rules: IndexBreakerRules | None = None,
        clock: Clock | None = None,
    ) -> None:
        symbols = [li.symbol for li in listings]
        if len(set(symbols)) != len(symbols):
            raise ValueError("symbols must be unique")
        self.clock = clock or Clock()
        self.ledger = Ledger()
        self.listings = {li.symbol: li for li in listings}
        self.markets: dict[str, Market] = {
            li.symbol: Market(li.initial_price, li.symbol, li.rules, self.clock, self.ledger)
            for li in listings
        }
        self.index_rules = index_rules
        self._index_base = sum(li.shares_outstanding * li.initial_price for li in listings)
        self.index_prev_close = 100.0
        self.index_stages_fired: set[float] = set()
        self.index_halts: list[tuple[int, float]] = []  # (step, threshold)
        self.index_auction_end: int | None = None
        self.t = 0

    # ------------------------------------------------------------------
    # Accounts
    # ------------------------------------------------------------------

    def register(self, name: str, cash: int = 0, positions: dict[str, int] | None = None) -> None:
        self.ledger.register(name, cash)
        for symbol, qty in (positions or {}).items():
            self.ledger.positions[name][symbol] = qty

    def cash(self, name: str) -> int:
        return self.ledger.cash[name]

    def position(self, name: str, symbol: str) -> int:
        return self.ledger.positions[name].get(symbol, 0)

    def transfer_external(self, name: str, amount: int) -> None:
        self.ledger.transfer_external(name, amount)

    def __getitem__(self, symbol: str) -> Market:
        return self.markets[symbol]

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def index_level(self) -> float:
        cap = sum(
            li.shares_outstanding * self.markets[s].last_price for s, li in self.listings.items()
        )
        return 100.0 * cap / self._index_base

    # ------------------------------------------------------------------
    # Step hooks
    # ------------------------------------------------------------------

    def begin_step(self, t: int) -> None:
        self.t = t
        for m in self.markets.values():
            m.t = t
        if self.clock.is_day_start(t):
            if t > 0:
                self.index_prev_close = self.index_level()
            self.index_stages_fired = set()
            for m in self.markets.values():
                m.start_day()
        for m in self.markets.values():
            m.maybe_end_auction()

    def index_halted(self, t: int) -> bool:
        return self.index_auction_end is not None and t < self.index_auction_end

    def end_step(self, t: int) -> None:
        if self.index_rules is None:
            return
        fall = 1.0 - self.index_level() / self.index_prev_close
        for threshold in self.index_rules.thresholds:
            if fall >= threshold and threshold not in self.index_stages_fired:
                self.index_stages_fired.add(threshold)
                self.index_halts.append((t, threshold))
                steps = self.clock.minutes_to_steps(self.index_rules.total_minutes())
                self.index_auction_end = t + steps
                for m in self.markets.values():
                    m.start_auction(steps)
                break  # one stage per step; the next stage can fire after reopening


# ---------------------------------------------------------------------------
# Agents and simulation loop
# ---------------------------------------------------------------------------


class ExchangeAgent(Protocol):
    name: str

    def act(self, t: int, exchange: Exchange, rng: np.random.Generator) -> None: ...


class SingleAssetAgent(Protocol):
    name: str

    def act(self, t: int, market: Market, rng: np.random.Generator) -> None: ...


class OnAsset:
    """Run a single-asset agent (noise trader, market maker...) on one symbol"""

    def __init__(self, symbol: str, agent: SingleAssetAgent) -> None:
        self.symbol = symbol
        self.agent = agent
        self.name = agent.name

    def act(self, t: int, exchange: Exchange, rng: np.random.Generator) -> None:
        self.agent.act(t, exchange[self.symbol], rng)


@dataclass(frozen=True, slots=True)
class ExchangeSnapshot:
    t: int
    index: float
    index_halted: bool  # an index circuit breaker auction is running
    assets: dict[str, Snapshot]
    halted: dict[str, bool]


@dataclass
class ExchangeSimulation:
    exchange: Exchange
    agents: list[ExchangeAgent]
    seed: int
    t: int = 0
    history: list[ExchangeSnapshot] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        names = [a.name for a in self.agents]
        if len(set(names)) != len(names):
            raise ValueError("agent names must be unique")
        for name in names:
            if name not in self.exchange.ledger.cash:
                self.exchange.register(name)

    def step(self) -> ExchangeSnapshot:
        ex = self.exchange
        n_before = {s: len(m.book.trades) for s, m in ex.markets.items()}
        ex.begin_step(self.t)
        for i in self.rng.permutation(len(self.agents)):
            self.agents[i].act(self.t, ex, self.rng)
        ex.end_step(self.t)

        snap = ExchangeSnapshot(
            t=self.t,
            index=ex.index_level(),
            index_halted=ex.index_halted(self.t),
            assets={
                s: snapshot_of(m, self.t, m.book.trades[n_before[s] :])
                for s, m in ex.markets.items()
            },
            halted={s: m.halted for s, m in ex.markets.items()},
        )
        self.history.append(snap)
        self.t += 1
        return snap

    def run(self, n_steps: int) -> list[ExchangeSnapshot]:
        for _ in range(n_steps):
            self.step()
        return self.history
