"""Random redemption runs: the varied scenarios an early-warning model learns from (Q4)

A real run is not one big redemption: investors leave in waves over hours. Each
scenario is drawn from its own seed:
    - with probability p_calm there is no run at all (the negatives)
    - otherwise 1-6 redemption waves, each 5-20% of units, starting on day 7 or 8,
      20-200 steps apart
    - with probability 0.3 a regulatory cap (8% of free float) hits with the first wave
    - the fund dumps (Immediate, probability 0.7) or sells each wave over half a day (TWAP)
Everything comes from the scenario seed, so a scenario is exactly reproducible
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from contagionsim.agents import FundShock, PortfolioFund
from contagionsim.execution import Immediate, Linear, Schedule
from contagionsim.market import ExchangeSimulation
from contagionsim.scenarios import MiniBistConfig, mini_bist

DAY = 1000


@dataclass(frozen=True, slots=True)
class RunScenario:
    seed: int
    wave_times: tuple[int, ...]
    wave_sizes: tuple[float, ...]
    cap: bool
    twap: bool
    days: int

    @property
    def is_calm(self) -> bool:
        return not self.wave_times

    def shocks(self) -> list[FundShock]:
        out = [
            FundShock(t, redeem_fraction=f)
            for t, f in zip(self.wave_times, self.wave_sizes, strict=True)
        ]
        if self.cap and self.wave_times:
            out.append(FundShock(self.wave_times[0], cap_fraction=0.08))
        return out

    def schedule(self) -> Schedule:
        return Linear(DAY // 2) if self.twap else Immediate()


def draw_scenario(seed: int, p_calm: float = 0.3, days: int = 10) -> RunScenario:
    # A separate generator: the scenario draw never touches the market's random stream
    rng = np.random.default_rng([seed, 2026])
    calm = rng.random() < p_calm
    cap = bool(rng.random() < 0.3)
    twap = bool(rng.random() < 0.3)
    if calm:
        return RunScenario(seed, (), (), False, twap, days)
    n_waves = int(rng.integers(1, 7))
    t = int(rng.integers(7 * DAY + 50, 8 * DAY + 800))
    times, sizes = [], []
    for _ in range(n_waves):
        times.append(t)
        sizes.append(round(float(rng.uniform(0.05, 0.20)), 3))
        t += int(rng.integers(20, 201))
    return RunScenario(seed, tuple(times), tuple(sizes), cap, twap, days)


def simulate(scenario: RunScenario) -> tuple[ExchangeSimulation, PortfolioFund]:
    config = dataclasses.replace(MiniBistConfig(), schedule=scenario.schedule())
    sim, fund = mini_bist(scenario.seed, scenario.shocks(), config)
    sim.run(scenario.days * DAY)
    return sim, fund
