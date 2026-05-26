"""Visualisation helpers for the ticket-routing simulation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Colour palette per strategy name (fallback: tab10)
_COLOURS: dict[str, str] = {
    "ts":           "#f77f00",   # orange — Beta TS
    "logistic":     "#4361ee",   # blue   — Logistic TS (no time)
    "logistic_full":"#7209b7",   # purple — Logistic TS (full)
    "round_robin":  "#aaaaaa",   # grey
    "oracle":       "#2dc653",   # green
}
_LABELS: dict[str, str] = {
    "ts":           "Beta TS",
    "logistic":     "Logistic TS",
    "logistic_full":"Logistic TS (full)",
    "round_robin":  "Round-robin",
    "oracle":       "Oracle",
}


def _colour(key: str, idx: int) -> str:
    if key in _COLOURS:
        return _COLOURS[key]
    tab10 = plt.cm.tab10.colors  # type: ignore[attr-defined]
    return tab10[idx % len(tab10)]


def _label(key: str) -> str:
    return _LABELS.get(key, key)


def plot_cumulative_fcr(results: dict, save_path: str | Path | None = None) -> None:
    """Plot cumulative FCR over time for all strategies in results.

    Args:
        results: {strategy_name: {"cumulative_fcr": list[float], ...}}
        save_path: If provided, save figure instead of displaying.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, (key, data) in enumerate(results.items()):
        if "cumulative_fcr" not in data:
            continue
        fcr = data["cumulative_fcr"]
        x = range(1, len(fcr) + 1)
        ls = "--" if key == "oracle" else "-"
        lw = 1.5 if key == "oracle" else 2
        ax.plot(x, fcr, color=_colour(key, idx), linestyle=ls, lw=lw, label=_label(key))

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
    """Plot cumulative regret vs Oracle for strategies that have a 'regret' key.

    Args:
        results: {strategy_name: {"regret": list[float], ...}}
        save_path: If provided, save figure instead of displaying.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, (key, data) in enumerate(results.items()):
        if "regret" not in data:
            continue
        cum_regret = np.cumsum(data["regret"])
        x = range(1, len(cum_regret) + 1)
        ax.plot(x, cum_regret, color=_colour(key, idx), lw=2, label=_label(key))

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
    assignments: dict[str, dict[str, int]],
    save_path: str | Path | None = None,
    title: str = "Agent Traffic Distribution by Category",
) -> None:
    """Stacked bar chart of agent ticket traffic per category.

    Args:
        assignments: {agent_id: {category: count}}
        save_path: If provided, save figure instead of displaying.
        title: Chart title.
    """
    categories = ["billing", "tech", "complaint", "onboard"]
    agents = list(assignments.keys())
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(agents))
    bottoms = np.zeros(len(agents))

    for i, cat in enumerate(categories):
        values = np.array([assignments[a].get(cat, 0) for a in agents], dtype=float)
        ax.bar(x, values, 0.6, bottom=bottoms, label=cat.capitalize(), color=colours[i])
        bottoms += values

    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=15)
    ax.set_ylabel("Number of tickets")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)
