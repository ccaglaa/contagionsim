# ContagionSim

An agent-based limit order book simulator of how a fund concentrated in illiquid stocks can turn a redemption run into a cross-asset fire sale. Inspired by the 2026 Turkish investment fund crisis. The model is stylized and does not represent any real fund or person.

**Status:** work in progress. M1 (matching engine), M2 (baseline market) and M3 (fund build-up and forced unwind) are done.

## Questions

1. **When does it break?** At what concentration and redemption shock does an orderly market tip into limit-down and circuit breakers?
2. **How should a forced seller liquidate?** Naive dumping vs pro-rata vs an optimal-execution schedule (Almgren-Chriss).
3. **What do risk models and rules miss?** A GARCH / historical VaR backtest through the crash, and a policy experiment on position caps, redemption gates and price-limit widths.
4. **Can you see it coming?** An early-warning classifier for limit-down events from order book signals, evaluated with temporal splits and precision-recall.

## Quick start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
python experiments/m2_baseline.py     # baseline market, saves results/m2_baseline.png
python experiments/m3_fund_unwind.py  # fund build-up and unwind, saves results/m3_fund_unwind.png
```

## Layout

```
src/contagionsim/orderbook/   limit order book, price-time priority matching
src/contagionsim/market/      Market, Exchange (multi-asset, BIST-style rules), Simulation loops
src/contagionsim/agents/      NoiseTrader, MarketMaker, ValueTrader, Fund
src/contagionsim/analysis/    stylized facts of returns
experiments/                  one script per figure
tests/                        pytest suite
```

## Design notes (M1)

- **Prices are integer ticks.** Floats make `0.1 + 0.2 != 0.3`, which breaks price comparisons and dict keys. Conversion to currency happens at the exchange layer.
- **Book structure:** per side, a dict `price -> deque` (FIFO time priority within a level) plus a sorted list of active prices (best price lookup). A dict `order_id -> Order` makes cancel O(1) to locate.
- **Sequence numbers, not timestamps,** define time priority, so seeded runs are exactly reproducible.
- **Trades execute at the resting order's price.**
- **Market orders never rest.** The unfilled remainder is reported as cancelled.
- **Bad input raises; cancelling an unknown order returns `False`.** A zero-quantity order is a caller bug; cancelling an order that was just filled is normal in a simulation.
- **Invariants** (never crossed, no empty levels, index consistent, FIFO) are checked after every step of a randomized stress test.

## Design notes (M2)

- **Agents go through `Market`, never the book directly.** `Market` settles every trade into both accounts, so cash and shares are conserved by construction (tested).
- **One seeded `numpy` Generator** drives all randomness, and the agent order is reshuffled every step, so no agent gets a permanent queue advantage.
- **Noise traders** send Pareto-sized orders (many small, a few huge, like real order flow). This is what produces fat-tailed returns.
- **Market maker** quotes around a reservation price `mid - skew * inventory` with a half-spread that widens with EWMA volatility (a simplified Avellaneda-Stoikov rule), stops quoting the side that would breach its inventory limit, and never crosses the book (post-only).
- **Baseline result (seed 0, 20k steps):** the book is two-sided 99.9% of the time, the mean spread is 2.5 ticks, and 20-step returns have excess kurtosis of about 33 (fat tails). The market maker takes part in about 11% of the volume, stays profitable, and keeps its inventory within a few hundred shares.
- **Known gap:** volatility clustering is weak (acf of |r| about 0.04), because noise traders act independently of the past. Feedback agents (fund flows, momentum retail investors) come in M3.

## Design notes (M3)

- **Fund:** NAV = cash - redemptions owed + shares x price. Investors chase performance: every 100 steps, flow = 2% + 2 x (return over the last 5 periods) + noise, capped at +/-10% of NAV.
- **Reflexive build-up:** the fund invests inflows in its one stock, which pushes the price up. NAV rises and attracts more inflows. The fund's own limit is 25% of free float.
- **First-mover advantage:** redeemers are paid at today's NAV, and the selling needed to pay them happens afterwards, so the price impact lands on the investors who stay (tested).
- **Value traders** buy below and sell above a fundamental value, with limited capacity. Short selling is much more limited than buying (500 vs 5,000 shares each), as on BIST. This is why the price can be pushed far above value.
- **Shock:** a regulator caps holdings at 8% of free float, and 30% of units are redeemed. The fund sells naively, all at once with market orders.
- **Result (seed 0):** the build-up pushes the price 29% above value. The forced sale of 17,000 shares crashes it to 23% *below* value, a fire-sale discount: value buyers hit their full 25,000-share capacity and cannot absorb more. Remaining investors lose 13% of NAV. Without the shock, the price stays inflated.
- **Known gaps:** without price limits the crash happens within a single step. Spreads barely widen, because value traders refill the book. Both are addressed in M4 (price limits, circuit breaker, multi-asset contagion).

## Exchange rules (M4, part 1)

Modeled on Borsa Istanbul. Every value is a parameter in `market/rules.py`.

| Segment | Daily price limit | Stock circuit breaker | Auction after a halt |
|---|---|---|---|
| Yildiz Pazar | +/-20% | 10% | 5 min collect + 2 min match |
| Ana Pazar | +/-15% | 7.5% | 15 min collect + 2 min match |
| Alt Pazar | +/-10% | 5% | 15 min collect + 2 min match |

- **Daily limit:** measured from the base price (previous close). Orders priced outside the band are rejected, and market orders stop at the band edge, so a stock can be *locked* at limit-down.
- **Stock circuit breaker:** an order that would trade beyond the trigger, measured from the reference price, halts the stock and starts a call auction. The auction price becomes the new reference.
- **Call auction:** orders collect without matching, then everything executes at one price. That price maximizes volume, then minimizes the leftover imbalance, then sits closest to the reference price.
- **Index circuit breaker:** a capitalization-weighted index falling 5% or 7% below the previous close halts every stock for 20 + 5 + 5 minutes. Each stage fires once per day.
- **Time:** one trading day is 1,000 steps, about 2 steps per minute of an 8-hour session.
- **Simplifications:** market orders are refused during auctions. The 20-minute index halt also accepts orders, so it is one long collection period. There is no end-of-session rule.

Sources: [Borsa Istanbul tick-size announcement (6/11/2023), via Phillip Capital](https://phillipcapital.com.tr/Files/OfficialAnnouncement/225f6ec0-7684-44eb-ae43-35cb8926eff1.pdf); [Is Yatirim, trading rules notice (price limits and circuit breakers by segment)](https://www.isyatirim.com.tr/tr-tr/Resmi-Duyurular/Documents/İşlem%20Kuralları%20Bildirim%20Formu.pdf); [Midas, circuit breakers on Borsa Istanbul (updated 29/09/2026)](https://www.getmidas.com/destek/borsa-istanbul/bist-piyasa-bilgileri/borsa-istanbulda-devre-kesici-nedir); [Forbes Turkiye, index circuit breaker triggered twice (19/03/2025)](https://www.forbes.com.tr/ekonomi/borsada-islemler-devre-kesicinin-calismasiyla-ikinci-kez-gecici-olarak-durduruldu).
