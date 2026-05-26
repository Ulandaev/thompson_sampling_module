"""Simulation tests for Phase 2 contextual routing (3 tests)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ts_module.core.engine import TSEngine

CONFIG_BETA = Path(__file__).parent.parent.parent / "examples" / "ticket_routing" / "config.yaml"
CONFIG_LOGISTIC = Path(__file__).parent.parent.parent / "examples" / "ticket_routing" / "config_contextual.yaml"
CONFIG_LOGISTIC_FULL = Path(__file__).parent.parent.parent / "examples" / "ticket_routing" / "config_contextual_full.yaml"

# Ground-truth tables (same as generator.py — duplicated to keep tests self-contained)
TRUE_FCR = {
    "agent_maria":  {"billing": 0.72, "tech": 0.68, "complaint": 0.91, "onboard": 0.70},
    "agent_ivan":   {"billing": 0.81, "tech": 0.74, "complaint": 0.61, "onboard": 0.79},
    "agent_alexey": {"billing": 0.65, "tech": 0.88, "complaint": 0.59, "onboard": 0.82},
    "agent_olga":   {"billing": 0.78, "tech": 0.71, "complaint": 0.74, "onboard": 0.93},
    "agent_dmitry": {"billing": 0.70, "tech": 0.65, "complaint": 0.67, "onboard": 0.68},
}
CATEGORIES = ["billing", "tech", "complaint", "onboard"]
CLIENT_TIERS = ["smb", "mid", "enterprise"]
TIER_WEIGHTS = [0.5, 0.35, 0.15]
CATEGORY_PROBS = [0.35, 0.28, 0.22, 0.15]


def _run_engine(config_path: Path, tickets: list[dict], rng_fcr: np.random.Generator, ts_seed: int) -> float:
    engine = TSEngine.from_yaml(str(config_path))
    rng_ts = np.random.default_rng(ts_seed)
    total_fcr = 0.0
    for ticket in tickets:
        result = engine.decide(context=ticket, seed=int(rng_ts.integers(2**31)))
        agent = result.recommended_arm
        true_p = TRUE_FCR[agent][ticket["ticket_category"]]
        fcr = 1.0 if rng_fcr.random() < true_p else 0.0
        engine.feedback(result.session_id, [{"name": "fcr", "value": fcr}])
        total_fcr += fcr
    return total_fcr / len(tickets)


def _gen_tickets(n: int, rng: np.random.Generator, include_time: bool = False) -> list[dict]:
    tickets = []
    for _ in range(n):
        cat = str(rng.choice(CATEGORIES, p=CATEGORY_PROBS))
        tier = str(rng.choice(CLIENT_TIERS, p=TIER_WEIGHTS))
        t: dict = {"ticket_category": cat, "client_tier": tier}
        if include_time:
            t["time_of_day"] = float(rng.uniform(8.0, 20.0))
        tickets.append(t)
    return tickets


def test_logistic_beats_beta_algorithmic_advantage() -> None:
    """Logistic TS >= Beta TS after 1000 tickets on same data (no time feature).

    Tests algorithmic advantage: shared weights generalise across contexts
    while Beta maintains independent distributions per context key.
    """
    rng_tix = np.random.default_rng(42)
    tickets = _gen_tickets(1000, rng_tix, include_time=False)

    beta_fcr = _run_engine(CONFIG_BETA, tickets, np.random.default_rng(100), ts_seed=200)
    logistic_fcr = _run_engine(CONFIG_LOGISTIC, tickets, np.random.default_rng(101), ts_seed=201)

    assert logistic_fcr >= beta_fcr, (
        f"Logistic FCR ({logistic_fcr:.3f}) should be >= Beta FCR ({beta_fcr:.3f})"
    )


def test_logistic_full_learns_time_effect() -> None:
    """Full model learns that agent_maria has higher predicted FCR in morning than evening.

    This directly tests that time_of_day was learned as a useful feature,
    which is more robust than comparing global FCR (which can be noisy at 1000 tickets).
    """
    from examples.ticket_routing.generator import simulate_fcr_with_time  # noqa: PLC0415

    rng_tix = np.random.default_rng(77)
    tickets = _gen_tickets(1000, rng_tix, include_time=True)

    engine = TSEngine.from_yaml(str(CONFIG_LOGISTIC_FULL))
    rng_ts = np.random.default_rng(400)
    rng_fcr = np.random.default_rng(300)

    for ticket in tickets:
        result = engine.decide(context=ticket, seed=int(rng_ts.integers(2**31)))
        agent = result.recommended_arm
        fcr = simulate_fcr_with_time(agent, ticket, rng_fcr)
        engine.feedback(result.session_id, [{"name": "fcr", "value": fcr}])

    # agent_maria gets +0.12 bonus in morning (8-12) and -0.06 penalty in evening.
    # After 1000 tickets the model should reflect this in estimated_p.
    morning_ctx = {"ticket_category": "complaint", "client_tier": "smb", "time_of_day": 10.0}
    evening_ctx = {"ticket_category": "complaint", "client_tier": "smb", "time_of_day": 19.0}

    p_morning = engine.get_arm_state("agent_maria", morning_ctx)["estimated_p"]
    p_evening = engine.get_arm_state("agent_maria", evening_ctx)["estimated_p"]

    assert p_morning > p_evening, (
        f"agent_maria morning p ({p_morning:.3f}) should be > evening p ({p_evening:.3f}): "
        "time_of_day feature was not learned"
    )


def test_logistic_learns_time_preference() -> None:
    """In morning slot (8-12), agent_maria is top for complaints AND gets >25% share."""
    morning_tickets = []
    rng = np.random.default_rng(55)
    for _ in range(1000):
        tier = str(rng.choice(CLIENT_TIERS, p=TIER_WEIGHTS))
        morning_tickets.append({
            "ticket_category": "complaint",
            "client_tier": tier,
            "time_of_day": float(rng.uniform(8.0, 12.0)),
        })

    from examples.ticket_routing.generator import simulate_fcr_with_time  # noqa: PLC0415

    engine = TSEngine.from_yaml(str(CONFIG_LOGISTIC_FULL))
    rng_ts = np.random.default_rng(99)
    rng_fcr = np.random.default_rng(999)

    agent_counts: dict[str, int] = {}
    for ticket in morning_tickets:
        result = engine.decide(context=ticket, seed=int(rng_ts.integers(2**31)))
        agent = result.recommended_arm
        fcr = simulate_fcr_with_time(agent, ticket, rng_fcr)
        engine.feedback(result.session_id, [{"name": "fcr", "value": fcr}])
        agent_counts[agent] = agent_counts.get(agent, 0) + 1

    total = sum(agent_counts.values())
    shares = {a: agent_counts.get(a, 0) / total for a in ["agent_maria", "agent_ivan", "agent_alexey", "agent_olga", "agent_dmitry"]}

    top_agent = max(shares, key=lambda a: shares[a])
    assert top_agent == "agent_maria", f"Expected agent_maria as top, got {top_agent} (shares={shares})"
    assert shares["agent_maria"] > 0.25, f"agent_maria share {shares['agent_maria']:.3f} < 0.25"
