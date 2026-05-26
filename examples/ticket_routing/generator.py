"""Ticket generator and FCR simulator for the ticket-routing simulation."""

from __future__ import annotations

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


def generate_ticket(rng: np.random.Generator) -> dict:
    """Generate a single ticket context dict.

    Args:
        rng: Seeded NumPy random generator.

    Returns:
        Dict with "ticket_category" and "client_tier" keys.
    """
    category = rng.choice(CATEGORIES, p=CATEGORY_PROBS)
    tier = rng.choice(CLIENT_TIERS, p=TIER_WEIGHTS)
    return {"ticket_category": str(category), "client_tier": str(tier)}


def simulate_fcr(agent_id: str, ticket: dict, rng: np.random.Generator) -> float:
    """Simulate a first-contact-resolution outcome for an agent on a ticket.

    Args:
        agent_id: One of the five known agent IDs.
        ticket: Ticket context dict (must contain "ticket_category").
        rng: Seeded NumPy random generator.

    Returns:
        1.0 if resolved on first contact, 0.0 otherwise (Bernoulli trial).
    """
    true_p = TRUE_FCR[agent_id][ticket["ticket_category"]]
    return 1.0 if rng.random() < true_p else 0.0
