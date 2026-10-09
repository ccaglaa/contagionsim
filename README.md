# ContagionSim

An agent-based limit order book simulator of how a fund concentrated in illiquid stocks can turn a redemption run into a cross-asset fire sale. Inspired by the 2026 Turkish investment fund crisis. The model is stylized and does not represent any real fund or person.

**Status:** work in progress. M1 (matching engine) and M2 (baseline market: noise traders + market maker) are done.

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
```

## Layout

```
src/contagionsim/orderbook/   limit order book, price-time priority matching
src/contagionsim/market/      Market (settlement, accounts), Simulation loop, Snapshot
src/contagionsim/agents/      NoiseTrader, MarketMaker
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
