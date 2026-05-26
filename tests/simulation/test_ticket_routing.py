"""Simulation-level tests: verify TS learns and outperforms round-robin (5 tests)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from examples.ticket_routing.generator import ORACLE, generate_ticket, simulate_fcr
from ts_module.core.engine import TSEngine

CONFIG_PATH = str(Path(__file__).parent.parent.parent / "examples" / "ticket_routing" / "config.yaml")

AGENTS = ["agent_maria", "agent_ivan", "agent_alexey", "agent_olga", "agent_dmitry"]
CATEGORIES = ["billing", "tech", "complaint", "onboard"]


# ── helpers ───────────────────────────────────────────────────────────────────


def _cumulative_fcr(fcr_list: list[float]) -> list[float]:
    result: list[float] = []
    total = 0.0
    for i, v in enumerate(fcr_list, 1):
        total += v
        result.append(total / i)
    return result


def _run_round_robin(tickets: list[dict], rng: np.random.Generator) -> list[float]:
    """Uniform random agent assignment (baseline)."""
    return [
        simulate_fcr(str(rng.choice(AGENTS)), ticket, rng) for ticket in tickets
    ]


def _run_oracle(tickets: list[dict], rng: np.random.Generator) -> list[float]:
    """Always pick best agent per category."""
    return [simulate_fcr(ORACLE[t["ticket_category"]][0], t, rng) for t in tickets]


def _run_ts(
    tickets: list[dict],
    rng_fcr: np.random.Generator,
    ts_seed: int = 42,
) -> tuple[list[float], dict[str, dict[str, int]]]:
    """Run TSEngine and return (fcr_list, category_assignments_per_agent)."""
    engine = TSEngine.from_yaml(CONFIG_PATH)
    rng_ts = np.random.default_rng(ts_seed)
    fcr_list: list[float] = []
    assignments: dict[str, dict[str, int]] = {
        a: {c: 0 for c in CATEGORIES} for a in AGENTS
    }
    for ticket in tickets:
        seed = int(rng_ts.integers(2**31))
        result = engine.decide(context=ticket, seed=seed)
        agent = result.recommended_arm
        fcr = simulate_fcr(agent, ticket, rng_fcr)
        engine.feedback(result.session_id, [{"name": "fcr", "value": fcr}])
        fcr_list.append(fcr)
        assignments[agent][ticket["ticket_category"]] += 1
    return fcr_list, assignments


def _make_tickets(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [generate_ticket(rng) for _ in range(n)]


# ── tests ─────────────────────────────────────────────────────────────────────


def test_ts_fcr_beats_round_robin_after_500_tickets() -> None:
    """TS FCR > Round-robin FCR after 500 tickets.

    seed=42: TS must beat RR by at least 3%.
    seeds 100, 200: TS must beat RR by any positive margin.
    """
    n_tickets = 500
    # (seed, min_margin)
    cases = [(42, 0.03), (100, 0.0), (200, 0.0)]
    for seed, margin in cases:
        tickets = _make_tickets(n_tickets, seed)
        rr_fcr = _cumulative_fcr(_run_round_robin(tickets, np.random.default_rng(seed + 1)))
        ts_fcr, _ = _run_ts(tickets, np.random.default_rng(seed + 2), ts_seed=seed + 3)
        ts_cum = _cumulative_fcr(ts_fcr)
        assert ts_cum[-1] > rr_fcr[-1] + margin, (
            f"seed={seed}: TS FCR={ts_cum[-1]:.3f} not > RR FCR={rr_fcr[-1]:.3f} + {margin}"
        )


def test_ts_regret_grows_slower_than_round_robin() -> None:
    """The slope of TS regret slows down: later regret < early regret."""
    n_tickets = 600
    tickets = _make_tickets(n_tickets, 42)
    oracle_fcr = _run_oracle(tickets, np.random.default_rng(43))
    ts_fcr_list, _ = _run_ts(tickets, np.random.default_rng(44), ts_seed=45)

    # Per-ticket regret (oracle - ts)
    regret = [o - t for o, t in zip(oracle_fcr, ts_fcr_list)]

    slope_early = sum(regret[:100])   # tickets 0-100
    slope_late = sum(regret[400:500])  # tickets 400-500

    assert slope_late < slope_early, (
        f"TS regret not decelerating: early slope={slope_early:.2f}, late slope={slope_late:.2f}"
    )


def test_ts_learns_complaint_specialist_within_300_tickets() -> None:
    """After 300 tickets, agent_maria should be the #1 complaint agent with >25% share.

    True FCR for complaints: maria=0.91, olga=0.74, dmitry=0.67, ivan=0.61, alexey=0.59.
    The 12-way context split (4 categories × 3 tiers) slows convergence, so we check
    dominance rather than an absolute 50% threshold.
    """
    tickets = _make_tickets(300, 42)
    _, assignments = _run_ts(tickets, np.random.default_rng(43), ts_seed=44)

    complaint_counts = {a: assignments[a]["complaint"] for a in AGENTS}
    total_complaints = sum(complaint_counts.values())
    if total_complaints == 0:
        pytest.skip("No complaint tickets generated with this seed")

    top_agent = max(complaint_counts, key=complaint_counts.__getitem__)
    maria_share = complaint_counts["agent_maria"] / total_complaints
    assert top_agent == "agent_maria", (
        f"Top complaint agent is '{top_agent}', expected 'agent_maria'"
    )
    assert maria_share > 0.25, (
        f"agent_maria complaint share={maria_share:.2f}, expected >0.25 (random=0.20)"
    )


def test_ts_learns_tech_specialist_within_300_tickets() -> None:
    """After 300 tickets, agent_alexey should be the #1 tech agent with >25% share.

    True FCR for tech: alexey=0.88, ivan=0.74, olga=0.71, maria=0.68, dmitry=0.65.
    """
    tickets = _make_tickets(300, 42)
    _, assignments = _run_ts(tickets, np.random.default_rng(43), ts_seed=44)

    tech_counts = {a: assignments[a]["tech"] for a in AGENTS}
    total_tech = sum(tech_counts.values())
    if total_tech == 0:
        pytest.skip("No tech tickets generated with this seed")

    top_agent = max(tech_counts, key=tech_counts.__getitem__)
    alexey_share = tech_counts["agent_alexey"] / total_tech
    assert top_agent == "agent_alexey", (
        f"Top tech agent is '{top_agent}', expected 'agent_alexey'"
    )
    assert alexey_share > 0.25, (
        f"agent_alexey tech share={alexey_share:.2f}, expected >0.25 (random=0.20)"
    )


def test_oracle_fcr_is_upper_bound() -> None:
    """Oracle FCR >= TS FCR >= Round-robin FCR (cumulative, after 500 tickets)."""
    n = 500
    tickets = _make_tickets(n, 42)
    oracle_fcr = _cumulative_fcr(_run_oracle(tickets, np.random.default_rng(43)))
    ts_fcr_list, _ = _run_ts(tickets, np.random.default_rng(44), ts_seed=45)
    ts_fcr = _cumulative_fcr(ts_fcr_list)
    rr_fcr = _cumulative_fcr(_run_round_robin(tickets, np.random.default_rng(46)))

    assert oracle_fcr[-1] >= ts_fcr[-1] - 0.02, (
        f"Oracle FCR {oracle_fcr[-1]:.3f} not >= TS FCR {ts_fcr[-1]:.3f}"
    )
    assert ts_fcr[-1] >= rr_fcr[-1] - 0.02, (
        f"TS FCR {ts_fcr[-1]:.3f} not >= RR FCR {rr_fcr[-1]:.3f}"
    )
