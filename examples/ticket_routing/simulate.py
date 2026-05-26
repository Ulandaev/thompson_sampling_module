"""Ticket-routing simulation comparing Beta TS, Logistic TS, Round-robin, and Oracle."""

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
    generate_ticket_with_time,
    simulate_fcr,
    simulate_fcr_with_time,
)
from examples.ticket_routing.visualize import (  # noqa: E402
    plot_agent_traffic,
    plot_cumulative_fcr,
    plot_regret,
)
from ts_module.core.engine import TSEngine  # noqa: E402

CONFIG_BETA = Path(__file__).parent / "config.yaml"
CONFIG_LOGISTIC = Path(__file__).parent / "config_contextual.yaml"
CONFIG_LOGISTIC_FULL = Path(__file__).parent / "config_contextual_full.yaml"

AGENTS = ["agent_maria", "agent_ivan", "agent_alexey", "agent_olga", "agent_dmitry"]
CATEGORIES = ["billing", "tech", "complaint", "onboard"]


def _empty_assignments() -> dict[str, dict[str, int]]:
    return {a: {c: 0 for c in CATEGORIES} for a in AGENTS}


def _cumulative_fcr(fcr_list: list[float]) -> list[float]:
    result: list[float] = []
    total = 0.0
    for i, v in enumerate(fcr_list, 1):
        total += v
        result.append(total / i)
    return result


# ── strategies ─────────────────────────────────────────────────────────────


def run_round_robin(tickets: list[dict], rng: np.random.Generator) -> dict:
    """Assign tickets uniformly at random (baseline)."""
    fcr_list: list[float] = []
    assignments = _empty_assignments()
    for ticket in tickets:
        agent = str(rng.choice(AGENTS))
        fcr = simulate_fcr(agent, ticket, rng)
        fcr_list.append(fcr)
        assignments[agent][ticket["ticket_category"]] += 1
    return {"fcr": fcr_list, "assignments": assignments}


def run_oracle(tickets: list[dict], rng: np.random.Generator, use_time: bool = False) -> dict:
    """Always pick the best agent per category (theoretical upper bound)."""
    fcr_list: list[float] = []
    assignments = _empty_assignments()
    for ticket in tickets:
        agent, _ = ORACLE[ticket["ticket_category"]]
        fcr = simulate_fcr_with_time(agent, ticket, rng) if use_time else simulate_fcr(agent, ticket, rng)
        fcr_list.append(fcr)
        assignments[agent][ticket["ticket_category"]] += 1
    return {"fcr": fcr_list, "assignments": assignments}


def run_ts(
    config_path: Path,
    tickets: list[dict],
    rng_fcr: np.random.Generator,
    ts_seed: int | None = None,
    use_time: bool = False,
) -> dict:
    """Run a TSEngine strategy (works for Beta, Logistic, or Linear config)."""
    engine = TSEngine.from_yaml(str(config_path))
    fcr_list: list[float] = []
    assignments = _empty_assignments()
    rng_ts = np.random.default_rng(ts_seed) if ts_seed is not None else None

    for ticket in tickets:
        decision_seed: int | None = None
        if rng_ts is not None:
            decision_seed = int(rng_ts.integers(2**31))

        result = engine.decide(context=ticket, seed=decision_seed)
        agent = result.recommended_arm
        fcr = simulate_fcr_with_time(agent, ticket, rng_fcr) if use_time else simulate_fcr(agent, ticket, rng_fcr)
        engine.feedback(result.session_id, [{"name": "fcr", "value": fcr}])
        fcr_list.append(fcr)
        assignments[agent][ticket["ticket_category"]] += 1

    return {"fcr": fcr_list, "assignments": assignments}


# ── display helpers ─────────────────────────────────────────────────────────


def _print_checkpoint_table(
    console: Console,
    title: str,
    checkpoints: list[int],
    n_tickets: int,
    strategies: dict[str, list[float]],
) -> None:
    table = Table(title=title)
    table.add_column("Tickets", justify="right")
    for name in strategies:
        table.add_column(name, justify="center")

    for cp in checkpoints:
        idx = min(cp, n_tickets) - 1
        row = [str(cp)] + [f"{cum[idx]:.3f}" for cum in strategies.values()]
        table.add_row(*row)

    console.print(table)


def _print_specialization_table(
    console: Console, title: str, assignments: dict[str, dict[str, int]]
) -> None:
    table = Table(title=title)
    table.add_column("Agent", style="cyan")
    for cat in CATEGORIES:
        table.add_column(cat.capitalize(), justify="center")
    table.add_column("Total", justify="right", style="bold")

    for agent in AGENTS:
        total = sum(assignments[agent].values())
        row: list[str] = [agent]
        for cat in CATEGORIES:
            pct = assignments[agent][cat] / total * 100 if total > 0 else 0.0
            row.append(f"{pct:.0f}%")
        row.append(str(total))
        table.add_row(*row)

    console.print(table)


# ── main ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Ticket routing TS simulation (Phase 2)")
    parser.add_argument("--tickets", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--save-results", action="store_true")
    args = parser.parse_args()

    n = args.tickets
    s = args.seed
    console = Console()

    # ── Experiment 1: no time effect — isolate algorithmic advantage ──────
    console.rule("[bold blue]Experiment 1: No time effect (Beta TS vs Logistic TS)")
    rng_tix1 = np.random.default_rng(s)
    tickets1 = [generate_ticket(rng_tix1) for _ in range(n)]

    rr1 = run_round_robin(tickets1, np.random.default_rng(s + 1))
    oracle1 = run_oracle(tickets1, np.random.default_rng(s + 2), use_time=False)
    beta1 = run_ts(CONFIG_BETA, tickets1, np.random.default_rng(s + 3), ts_seed=s + 4)
    logistic1 = run_ts(CONFIG_LOGISTIC, tickets1, np.random.default_rng(s + 5), ts_seed=s + 6)

    _print_checkpoint_table(
        console,
        title="FCR by tickets (no time effect)",
        checkpoints=[500, 1000, n],
        n_tickets=n,
        strategies={
            "Round-robin": _cumulative_fcr(rr1["fcr"]),
            "Beta TS": _cumulative_fcr(beta1["fcr"]),
            "Logistic TS": _cumulative_fcr(logistic1["fcr"]),
            "Oracle": _cumulative_fcr(oracle1["fcr"]),
        },
    )
    _print_specialization_table(console, "Agent specialization — Logistic TS (no time)", logistic1["assignments"])

    # ── Experiment 2: with time effect — feature value ────────────────────
    console.rule("[bold green]Experiment 2: With time effect (feature advantage)")
    rng_tix2 = np.random.default_rng(s + 10)
    tickets2 = [generate_ticket_with_time(rng_tix2) for _ in range(n)]

    oracle2 = run_oracle(tickets2, np.random.default_rng(s + 12), use_time=True)
    beta2 = run_ts(CONFIG_BETA, tickets2, np.random.default_rng(s + 13), ts_seed=s + 14, use_time=True)
    logistic2_no = run_ts(CONFIG_LOGISTIC, tickets2, np.random.default_rng(s + 15), ts_seed=s + 16, use_time=True)
    logistic2_full = run_ts(CONFIG_LOGISTIC_FULL, tickets2, np.random.default_rng(s + 17), ts_seed=s + 18, use_time=True)

    _print_checkpoint_table(
        console,
        title="FCR by tickets (with time effect)",
        checkpoints=[500, 1000, n],
        n_tickets=n,
        strategies={
            "Beta TS": _cumulative_fcr(beta2["fcr"]),
            "Logistic (no time)": _cumulative_fcr(logistic2_no["fcr"]),
            "Logistic (full)": _cumulative_fcr(logistic2_full["fcr"]),
            "Oracle": _cumulative_fcr(oracle2["fcr"]),
        },
    )
    _print_specialization_table(console, "Agent specialization — Logistic full (with time)", logistic2_full["assignments"])

    # ── optional plots ───────────────────────────────────────────────────
    if args.show_plots or args.save_results:
        save_dir = Path("results") if args.save_results else None
        if save_dir:
            save_dir.mkdir(exist_ok=True)

        def _sp(name: str) -> Path | None:
            return save_dir / name if save_dir else None

        # Experiment 1: Beta TS vs Logistic TS vs Round-robin (no time effect)
        rr_cum1 = _cumulative_fcr(rr1["fcr"])
        oracle_cum1 = _cumulative_fcr(oracle1["fcr"])
        beta_cum1 = _cumulative_fcr(beta1["fcr"])
        logistic_cum1 = _cumulative_fcr(logistic1["fcr"])

        exp1_data = {
            "round_robin": {"cumulative_fcr": rr_cum1, "regret": [o - r for o, r in zip(oracle_cum1, rr_cum1)]},
            "ts": {"cumulative_fcr": beta_cum1, "regret": [o - t for o, t in zip(oracle_cum1, beta_cum1)]},
            "logistic": {"cumulative_fcr": logistic_cum1, "regret": [o - t for o, t in zip(oracle_cum1, logistic_cum1)]},
            "oracle": {"cumulative_fcr": oracle_cum1},
        }
        plot_cumulative_fcr(exp1_data, save_path=_sp("fcr_exp1.png"))
        plot_regret(exp1_data, save_path=_sp("regret_exp1.png"))
        plot_agent_traffic(
            logistic1["assignments"],
            save_path=_sp("traffic_exp1.png"),
            title="Agent Traffic — Logistic TS (no time effect)",
        )

        # Experiment 2: Beta TS vs Logistic no-time vs Logistic full (with time effect)
        oracle_cum2 = _cumulative_fcr(oracle2["fcr"])
        beta_cum2 = _cumulative_fcr(beta2["fcr"])
        logistic_cum2_no = _cumulative_fcr(logistic2_no["fcr"])
        logistic_cum2_full = _cumulative_fcr(logistic2_full["fcr"])

        exp2_data = {
            "ts": {"cumulative_fcr": beta_cum2, "regret": [o - t for o, t in zip(oracle_cum2, beta_cum2)]},
            "logistic": {"cumulative_fcr": logistic_cum2_no, "regret": [o - t for o, t in zip(oracle_cum2, logistic_cum2_no)]},
            "logistic_full": {"cumulative_fcr": logistic_cum2_full, "regret": [o - t for o, t in zip(oracle_cum2, logistic_cum2_full)]},
            "oracle": {"cumulative_fcr": oracle_cum2},
        }
        plot_cumulative_fcr(exp2_data, save_path=_sp("fcr_exp2.png"))
        plot_regret(exp2_data, save_path=_sp("regret_exp2.png"))
        plot_agent_traffic(
            logistic2_no["assignments"],
            save_path=_sp("traffic_exp2_no.png"),
            title="Agent Traffic — Logistic TS (no time feature)",
        )
        plot_agent_traffic(
            logistic2_full["assignments"],
            save_path=_sp("traffic_exp2_full.png"),
            title="Agent Traffic — Logistic TS (full features with time)",
        )


if __name__ == "__main__":
    main()
