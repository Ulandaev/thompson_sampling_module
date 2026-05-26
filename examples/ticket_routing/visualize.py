"""Visualisation helpers for the ticket-routing simulation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_cumulative_fcr(results: dict, save_path: str | Path | None = None) -> None:
    """Plot cumulative FCR over time for all three strategies.

    Args:
        results: Dict with keys "ts", "round_robin", "oracle"; each must have
                 "cumulative_fcr" as a list of floats.
        save_path: If provided, save figure to this path instead of displaying.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    n = len(results["ts"]["cumulative_fcr"])
    x = range(1, n + 1)

    ax.plot(x, results["ts"]["cumulative_fcr"], color="orange", label="TS (our module)", lw=2)
    ax.plot(x, results["round_robin"]["cumulative_fcr"], color="gray", label="Round-robin", lw=2)
    ax.plot(
        x,
        results["oracle"]["cumulative_fcr"],
        color="green",
        linestyle="--",
        label="Oracle",
        lw=2,
    )

    ax.set_xlabel("Ticket number")
    ax.set_ylabel("Cumulative FCR")
    ax.set_title("Cumulative First Contact Resolution Rate")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)


def plot_regret(results: dict, save_path: str | Path | None = None) -> None:
    """Plot cumulative regret for TS and Round-robin vs Oracle.

    Args:
        results: Dict with keys "ts" and "round_robin"; each must have
                 "regret" as a per-ticket list of floats.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    n = len(results["ts"]["regret"])
    x = range(1, n + 1)

    ts_cumregret = np.cumsum(results["ts"]["regret"])
    rr_cumregret = np.cumsum(results["round_robin"]["regret"])

    ax.plot(x, ts_cumregret, color="orange", label="TS regret", lw=2)
    ax.plot(x, rr_cumregret, color="gray", label="Round-robin regret", lw=2)

    ax.set_xlabel("Ticket number")
    ax.set_ylabel("Cumulative regret vs Oracle")
    ax.set_title("Cumulative Regret vs Oracle")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)


def plot_agent_traffic(
    ts_assignments: dict[str, dict[str, int]],
    save_path: str | Path | None = None,
) -> None:
    """Plot stacked bar chart of agent ticket traffic per category.

    Args:
        ts_assignments: {agent_id: {category: count}} from the TS run.
        save_path: If provided, save figure to this path.
    """
    categories = ["billing", "tech", "complaint", "onboard"]
    agents = list(ts_assignments.keys())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(agents))
    bottoms = np.zeros(len(agents))

    for i, cat in enumerate(categories):
        values = np.array([ts_assignments[a].get(cat, 0) for a in agents], dtype=float)
        ax.bar(x, values, 0.6, bottom=bottoms, label=cat.capitalize(), color=colors[i])
        bottoms += values

    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=15)
    ax.set_ylabel("Number of tickets")
    ax.set_title("Agent Traffic Distribution by Category (TS)")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)
