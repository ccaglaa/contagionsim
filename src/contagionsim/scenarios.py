"""Ready-made market setups, shared by experiments and tests

mini_bist: one cornered small cap (Ana Pazar rules) + four liquid large caps
(Yildiz Pazar rules) forming a cap-weighted index with the BIST 100 circuit breaker,
plus a PortfolioFund that holds both

Calibration targets (M4, see README):
    - calm run: index daily volatility ~1%, no index circuit breaker, few stock halts
    - liquid stocks much deeper than the cornered stock (5x order sizes, market
      maker size and arbitrage capital)
    - fund large enough to own 25% of the cornered stock's float and a few days
      of the liquid stocks' volume
"""

from __future__ import annotations

from dataclasses import dataclass, field

from contagionsim.agents import (
    FundShock,
    MarketMaker,
    MarketMakerParams,
    NoiseTrader,
    NoiseTraderParams,
    PortfolioFund,
    PortfolioFundParams,
    ValueTrader,
    ValueTraderParams,
)
from contagionsim.market import (
    ANA,
    YILDIZ,
    Exchange,
    ExchangeAgent,
    ExchangeSimulation,
    IndexBreakerRules,
    Listing,
    OnAsset,
    SegmentRules,
)

FUNDAMENTAL = 1000  # ticks = 10.00 TRY, for every stock


@dataclass(frozen=True)
class MiniBistConfig:
    concentrated: str = "CORNER"
    liquid: tuple[str, ...] = ("L1", "L2", "L3", "L4")
    concentrated_rules: SegmentRules = ANA
    liquid_rules: SegmentRules = YILDIZ
    index_rules: IndexBreakerRules | None = field(default_factory=IndexBreakerRules)
    concentrated_shares: int = 200_000  # index weight; free float is in the fund params
    liquid_shares: int = 2_000_000
    liquid_depth: int = 5  # liquid stocks: x order size, MM size, arbitrage capital
    n_noise_concentrated: int = 8
    n_noise_liquid: int = 16
    n_value: int = 5
    fund: PortfolioFundParams = field(
        default_factory=lambda: PortfolioFundParams(
            initial_cash=30_000_000,  # 300k TRY at launch
            max_aum=250_000_000,  # soft close at 2.5M TRY
        )
    )


def mini_bist(
    seed: int,
    shocks: list[FundShock] | None = None,
    config: MiniBistConfig | None = None,
) -> tuple[ExchangeSimulation, PortfolioFund]:
    cfg = config or MiniBistConfig()
    listings = [
        Listing(cfg.concentrated, FUNDAMENTAL, cfg.concentrated_shares, cfg.concentrated_rules)
    ]
    listings += [Listing(s, FUNDAMENTAL, cfg.liquid_shares, cfg.liquid_rules) for s in cfg.liquid]
    exchange = Exchange(listings, cfg.index_rules)

    agents: list[ExchangeAgent] = []
    for sym in (cfg.concentrated, *cfg.liquid):
        liquid = sym != cfg.concentrated
        k = cfg.liquid_depth if liquid else 1
        noise = NoiseTraderParams(min_size=10 * k)
        n_noise = cfg.n_noise_liquid if liquid else cfg.n_noise_concentrated
        agents += [OnAsset(sym, NoiseTrader(f"noise_{sym}_{i}", noise)) for i in range(n_noise)]
        mm = MarketMakerParams(quote_size=100 * k, max_inventory=2000 * k)
        agents.append(OnAsset(sym, MarketMaker(f"mm_{sym}", mm)))
        if liquid:
            # Large caps: many arbitrageurs, shorting possible, act on 2% mispricing
            capital = 4000 * k
            value = ValueTraderParams(
                fundamental=FUNDAMENTAL,
                threshold=0.02,
                shares_per_pct=20 * k,
                max_long=capital,
                max_short=capital,
            )
        else:
            # Cornered small cap: short selling nearly impossible (see M3)
            value = ValueTraderParams(fundamental=FUNDAMENTAL)
        agents += [OnAsset(sym, ValueTrader(f"value_{sym}_{i}", value)) for i in range(cfg.n_value)]

    fund_params = cfg.fund
    if fund_params.concentrated != cfg.concentrated or fund_params.liquid != cfg.liquid:
        raise ValueError("fund symbols must match the listings")
    fund = PortfolioFund("fund", fund_params, shocks)
    agents.append(fund)
    return ExchangeSimulation(exchange, agents, seed=seed), fund
