"""Exchange rules, modeled on Borsa Istanbul (values as of 2025-2026, see README sources)

Everything here is a parameter, so experiments can change the rules (Q3 policy work)

Per-stock rules depend on the market segment:
    segment       daily limit   stock circuit breaker   auction after it
    Yildiz Pazar      20%              10%              5 min collect + 2 min match
    Ana Pazar         15%              7.5%             15 min collect + 2 min match
    Alt Pazar         10%              5%               15 min collect + 2 min match
Index circuit breaker (BIST 100): -5% and -7% from the previous close, each once
per day; all stocks halt 20 min, then 5 min collect + 5 min match

Simulation time: 1 trading day = steps_per_day steps, with steps_per_minute steps
per minute of real session time. The default 1000 steps / day is about 2 per minute
for an 8-hour session. A halt of 20 minutes is therefore 40 steps
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SegmentRules:
    name: str
    daily_limit: float  # +/- fraction of the base price (previous close)
    cb_trigger: float  # move from the circuit-breaker reference that halts the stock
    cb_collect_minutes: float
    cb_match_minutes: float

    def auction_minutes(self) -> float:
        return self.cb_collect_minutes + self.cb_match_minutes


YILDIZ = SegmentRules("Yildiz Pazar", daily_limit=0.20, cb_trigger=0.10,
                      cb_collect_minutes=5, cb_match_minutes=2)  # fmt: skip
ANA = SegmentRules("Ana Pazar", daily_limit=0.15, cb_trigger=0.075,
                   cb_collect_minutes=15, cb_match_minutes=2)  # fmt: skip
ALT = SegmentRules("Alt Pazar", daily_limit=0.10, cb_trigger=0.05,
                   cb_collect_minutes=15, cb_match_minutes=2)  # fmt: skip
NO_LIMITS = SegmentRules("No limits", daily_limit=10.0, cb_trigger=10.0,
                         cb_collect_minutes=0, cb_match_minutes=0)  # fmt: skip


@dataclass(frozen=True, slots=True)
class IndexBreakerRules:
    thresholds: tuple[float, ...] = (0.05, 0.07)  # falls from the previous close
    halt_minutes: float = 20
    collect_minutes: float = 5
    match_minutes: float = 5

    def total_minutes(self) -> float:
        return self.halt_minutes + self.collect_minutes + self.match_minutes


@dataclass(frozen=True, slots=True)
class Clock:
    steps_per_day: int = 1000
    steps_per_minute: float = 2.0

    def minutes_to_steps(self, minutes: float) -> int:
        return max(1, round(minutes * self.steps_per_minute))

    def day(self, t: int) -> int:
        return t // self.steps_per_day

    def is_day_start(self, t: int) -> bool:
        return t % self.steps_per_day == 0
