"""Ticket generator and FCR simulator for the ticket-routing simulation."""

from __future__ import annotations

from typing import Callable

import numpy as np

# ── ground-truth FCR per agent per category ──────────────────────────────────

TRUE_FCR: dict[str, dict[str, float]] = {
    "agent_maria": {
        "billing": 0.72,
        "tech": 0.68,
        "complaint": 0.91,
        "onboard": 0.70,
    },
    "agent_ivan": {
        "billing": 0.81,
        "tech": 0.74,
        "complaint": 0.61,
        "onboard": 0.79,
    },
    "agent_alexey": {
        "billing": 0.65,
        "tech": 0.88,
        "complaint": 0.59,
        "onboard": 0.82,
    },
    "agent_olga": {
        "billing": 0.78,
        "tech": 0.71,
        "complaint": 0.74,
        "onboard": 0.93,
    },
    "agent_dmitry": {
        "billing": 0.70,
        "tech": 0.65,
        "complaint": 0.67,
        "onboard": 0.68,
    },
}

# ── oracle: best agent per category ──────────────────────────────────────────

ORACLE: dict[str, tuple[str, float]] = {
    "billing": ("agent_ivan", 0.81),
    "tech": ("agent_alexey", 0.88),
    "complaint": ("agent_maria", 0.91),
    "onboard": ("agent_olga", 0.93),
}

# ── ticket distribution ───────────────────────────────────────────────────────

TICKET_DIST: dict[str, float] = {
    "billing": 0.35,
    "tech": 0.28,
    "complaint": 0.22,
    "onboard": 0.15,
}

CLIENT_TIERS: list[str] = ["smb", "mid", "enterprise"]
TIER_WEIGHTS: list[float] = [0.5, 0.35, 0.15]

CATEGORIES = list(TICKET_DIST.keys())
CATEGORY_PROBS = [TICKET_DIST[c] for c in CATEGORIES]


# ── time-of-day effects (Phase 2) ────────────────────────────────────────────

TIME_BONUS: dict[str, Callable[[float], float]] = {
    "agent_maria":  lambda t: +0.12 if 8 <= t <= 12 else -0.06,
    "agent_ivan":   lambda t: +0.08 if 9 <= t <= 13 else -0.04,
    "agent_alexey": lambda t: +0.10 if 14 <= t <= 18 else -0.05,
    "agent_olga":   lambda t: +0.08 if 10 <= t <= 15 else -0.04,
    "agent_dmitry": lambda t: +0.05 if 11 <= t <= 16 else -0.03,
}


def true_fcr_with_time(agent_id: str, category: str, time_of_day: float) -> float:
    """Base FCR adjusted by agent-specific time-of-day bonus."""
    base = TRUE_FCR[agent_id][category]
    bonus = TIME_BONUS[agent_id](time_of_day)
    return float(np.clip(base + bonus, 0.05, 0.98))


def generate_ticket(rng: np.random.Generator) -> dict:
    """Generate a ticket context dict (no time feature — Phase 1 compatible)."""
    category = rng.choice(CATEGORIES, p=CATEGORY_PROBS)
    tier = rng.choice(CLIENT_TIERS, p=TIER_WEIGHTS)
    return {"ticket_category": str(category), "client_tier": str(tier)}


def generate_ticket_with_time(rng: np.random.Generator) -> dict:
    """Generate a ticket context dict including a time_of_day continuous feature."""
    base = generate_ticket(rng)
    base["time_of_day"] = float(rng.uniform(8.0, 20.0))
    return base


def simulate_fcr(agent_id: str, ticket: dict, rng: np.random.Generator) -> float:
    """Simulate FCR outcome — uses base TRUE_FCR (no time effect)."""
    true_p = TRUE_FCR[agent_id][ticket["ticket_category"]]
    return 1.0 if rng.random() < true_p else 0.0


def simulate_fcr_with_time(agent_id: str, ticket: dict, rng: np.random.Generator) -> float:
    """Simulate FCR outcome with time-of-day effect applied."""
    true_p = true_fcr_with_time(agent_id, ticket["ticket_category"], ticket["time_of_day"])
    return 1.0 if rng.random() < true_p else 0.0
