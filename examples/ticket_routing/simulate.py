"""Ticket-routing simulation comparing Thompson Sampling, Round-robin, and Oracle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

# Allow running as a script from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.ticket_routing.generator import (  # noqa: E402
    ORACLE,
    generate_ticket,
    simulate_fcr,
)
from examples.ticket_routing.visualize import (  # noqa: E402
    plot_agent_traffic,
    plot_cumulative_fcr,
    plot_regret,
)
from ts_module.core.engine import TSEngine  # noqa: E402

CONFIG_PATH = Path(__file__).parent / "config.yaml"

AGENTS = ["agent_maria", "agent_ivan", "agent_alexey", "agent_olga", "agent_dmitry"]
CATEGORIES = ["billing", "tech", "complaint", "onboard"]


def _empty_assignments() -> dict[str, dict[str, int]]:
    return {a: {c: 0 for c in CATEGORIES} for a in AGENTS}


def _cumulative_fcr(fcr_list: list[float]) -> list[float]:
    """Convert a per-ticket FCR list to cumulative averages."""
    result: list[float] = []
    total = 0.0
    for i, v in enumerate(fcr_list, 1):
        total += v
        result.append(total / i)
    return result


# ── strategies ─────────────────────────────────────────────────────────────


def run_round_robin(
    tickets: list[dict],
    rng: np.random.Generator,
) -> dict:
    """Assign tickets uniformly at random (baseline)."""
    fcr_list: list[float] = []
    assignments = _empty_assignments()

    for ticket in tickets:
        agent = str(rng.choice(AGENTS))
        fcr = simulate_fcr(agent, ticket, rng)
        fcr_list.append(fcr)
        assignments[agent][ticket["ticket_category"]] += 1

    return {"fcr": fcr_list, "assignments": assignments}


def run_oracle(
    tickets: list[dict],
    rng: np.random.Generator,
) -> dict:
    """Always pick the best agent for each category (theoretical upper bound)."""
    fcr_list: list[float] = []
    assignments = _empty_assignments()

    for ticket in tickets:
        agent, _ = ORACLE[ticket["ticket_category"]]
        fcr = simulate_fcr(agent, ticket, rng)
        fcr_list.append(fcr)
        assignments[agent][ticket["ticket_category"]] += 1

    return {"fcr": fcr_list, "assignments": assignments}


def run_ts(
    tickets: list[dict],
    rng_fcr: np.random.Generator,
    ts_seed: int | None = None,
) -> dict:
    """Run Thompson Sampling strategy using TSEngine.

    Args:
        tickets: Pre-generated ticket stream.
        rng_fcr: RNG used to simulate FCR outcomes.
        ts_seed: Optional base seed for deterministic TS sampling.
                 Each decision derives a unique sub-seed from this.
    """
    engine = TSEngine.from_yaml(str(CONFIG_PATH))
    fcr_list: list[float] = []
    assignments = _empty_assignments()

    rng_ts = np.random.default_rng(ts_seed) if ts_seed is not None else None

    for ticket in tickets:
        decision_seed: int | None = None
        if rng_ts is not None:
            decision_seed = int(rng_ts.integers(2**31))

        result = engine.decide(context=ticket, seed=decision_seed)
        agent = result.recommended_arm
        fcr = simulate_fcr(agent, ticket, rng_fcr)
        engine.feedback(result.session_id, [{"name": "fcr", "value": fcr}])
        fcr_list.append(fcr)
        assignments[agent][ticket["ticket_category"]] += 1

    return {"fcr": fcr_list, "assignments": assignments}


# ── main ───────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the simulation script."""
    parser = argparse.ArgumentParser(description="Ticket routing TS simulation")
    parser.add_argument("--tickets", type=int, default=1500, help="Number of tickets to simulate")
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed")
    parser.add_argument("--show-plots", action="store_true", help="Display matplotlib plots")
    parser.add_argument("--save-results", action="store_true", help="Save CSV/PNG to results/")
    args = parser.parse_args()

    # Generate shared ticket stream
    rng_tickets = np.random.default_rng(args.seed)
    tickets = [generate_ticket(rng_tickets) for _ in range(args.tickets)]

    # Independent RNGs for FCR simulation per strategy
    rr_results = run_round_robin(tickets, np.random.default_rng(args.seed + 1))
    oracle_results = run_oracle(tickets, np.random.default_rng(args.seed + 2))
    ts_results = run_ts(
        tickets,
        rng_fcr=np.random.default_rng(args.seed + 3),
        ts_seed=args.seed + 4,
    )

    rr_cum = _cumulative_fcr(rr_results["fcr"])
    oracle_cum = _cumulative_fcr(oracle_results["fcr"])
    ts_cum = _cumulative_fcr(ts_results["fcr"])

    console = Console()

    # ── checkpoint summary ───────────────────────────────────────────────
    for cp in [500, 1000, args.tickets]:
        idx = min(cp, args.tickets) - 1
        console.print(
            f"Ticket {cp:4d}:  TS FCR={ts_cum[idx]:.3f} | "
            f"RR FCR={rr_cum[idx]:.3f} | "
            f"Oracle FCR={oracle_cum[idx]:.3f}"
        )

    console.print()

    # ── agent specialisation table ───────────────────────────────────────
    table = Table(title=f"Agent specialization after {args.tickets} tickets (TS)")
    table.add_column("Agent", style="cyan")
    for cat in CATEGORIES:
        table.add_column(cat.capitalize(), justify="center")
    table.add_column("Total", justify="right", style="bold")

    ts_assignments = ts_results["assignments"]
    for agent in AGENTS:
        total = sum(ts_assignments[agent].values())
        row: list[str] = [agent]
        for cat in CATEGORIES:
            count = ts_assignments[agent][cat]
            pct = count / total * 100 if total > 0 else 0.0
            row.append(f"{pct:.0f}%")
        row.append(str(total))
        table.add_row(*row)

    console.print(table)

    # ── optional plots ───────────────────────────────────────────────────
    if args.show_plots or args.save_results:
        per_ticket_oracle = [o - r for o, r in zip(oracle_cum, rr_cum)]
        per_ticket_ts = [o - t for o, t in zip(oracle_cum, ts_cum)]
        plot_data = {
            "ts": {"cumulative_fcr": ts_cum, "regret": per_ticket_ts},
            "round_robin": {"cumulative_fcr": rr_cum, "regret": per_ticket_oracle},
            "oracle": {"cumulative_fcr": oracle_cum},
        }
        save_dir = Path("results") if args.save_results else None
        if save_dir:
            save_dir.mkdir(exist_ok=True)
        plot_cumulative_fcr(plot_data, save_path=save_dir / "fcr.png" if save_dir else None)
        plot_regret(plot_data, save_path=save_dir / "regret.png" if save_dir else None)
        plot_agent_traffic(
            ts_assignments, save_path=save_dir / "traffic.png" if save_dir else None
        )


if __name__ == "__main__":
    main()
