"""Market (settlement and bookkeeping) and Simulation loop tests"""

from __future__ import annotations

import pytest

from contagionsim.agents import MarketMaker, NoiseTrader
from contagionsim.market import Market, Simulation
from contagionsim.orderbook import Side


def make_market() -> Market:
    m = Market(initial_price=1000)
    for name in ("alice", "bob", "carol"):
        m.register(name)
    return m


def test_trade_settles_cash_and_position_on_both_sides() -> None:
    m = make_market()
    m.submit_limit("alice", Side.SELL, 100, 1005)
    m.submit_market("bob", Side.BUY, 40)

    alice, bob = m.accounts["alice"], m.accounts["bob"]
    assert (alice.position, alice.cash) == (-40, 40 * 1005)
    assert (bob.position, bob.cash) == (40, -40 * 1005)
    assert m.last_price == 1005


def test_cash_and_shares_are_conserved() -> None:
    # Trading moves cash and shares between agents, never creates them
    m = make_market()
    m.submit_limit("alice", Side.SELL, 100, 1005)
    m.submit_limit("carol", Side.SELL, 50, 1006)
    m.submit_market("bob", Side.BUY, 130)
    m.submit_limit("alice", Side.BUY, 20, 1007)
    assert sum(a.cash for a in m.accounts.values()) == 0
    assert sum(a.position for a in m.accounts.values()) == 0


def test_reference_price_falls_back_to_last_trade() -> None:
    m = make_market()
    assert m.reference_price() == 1000.0  # empty book: initial price
    m.submit_limit("alice", Side.BUY, 10, 990)
    m.submit_limit("bob", Side.SELL, 10, 1000)
    assert m.reference_price() == 995.0  # two-sided: mid
    m.submit_market("carol", Side.BUY, 10)  # takes the only ask at 1000
    assert m.reference_price() == 1000.0  # one-sided: last trade


def test_cannot_cancel_someone_elses_order() -> None:
    m = make_market()
    oid = m.submit_limit("alice", Side.BUY, 10, 990).order_id
    with pytest.raises(PermissionError):
        m.cancel("bob", oid)
    assert m.cancel("alice", oid) is True


def test_unregistered_agent_cannot_trade() -> None:
    m = make_market()
    with pytest.raises(KeyError):
        m.submit_market("mallory", Side.BUY, 10)


def test_register_twice_raises() -> None:
    m = make_market()
    with pytest.raises(ValueError):
        m.register("alice")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def baseline_sim(seed: int, n_noise: int = 20) -> Simulation:
    agents = [NoiseTrader(f"noise{i}") for i in range(n_noise)]
    agents.append(MarketMaker("mm"))
    return Simulation(Market(initial_price=1000), agents, seed=seed)


def test_same_seed_gives_identical_run() -> None:
    a = baseline_sim(seed=42).run(500)
    b = baseline_sim(seed=42).run(500)
    assert a == b


def test_different_seeds_give_different_runs() -> None:
    a = baseline_sim(seed=1).run(500)
    b = baseline_sim(seed=2).run(500)
    assert a != b


def test_simulation_keeps_book_valid_and_conserves_cash() -> None:
    sim = baseline_sim(seed=7)
    for _ in range(1000):
        snap = sim.step()
        sim.market.book.check_invariants()
        assert snap.last_price > 0
    accounts = sim.market.accounts.values()
    assert sum(a.cash for a in accounts) == 0
    assert sum(a.position for a in accounts) == 0


def test_baseline_market_is_liquid() -> None:
    hist = baseline_sim(seed=3).run(2000)
    two_sided = sum(s.mid is not None for s in hist) / len(hist)
    assert two_sided > 0.95
    assert sum(s.volume for s in hist) > 0


def test_positions_are_recorded_every_step() -> None:
    sim = baseline_sim(seed=0, n_noise=3)
    sim.run(50)
    assert all(len(series) == 50 for series in sim.positions.values())


def test_duplicate_agent_names_rejected() -> None:
    with pytest.raises(ValueError):
        Simulation(Market(1000), [NoiseTrader("x"), NoiseTrader("x")], seed=0)
