"""Shared helpers for the generated phase reports."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, str):
        return value
    value = float(value)
    if not np.isfinite(value):
        return str(value)
    if value != 0 and (abs(value) >= 1e5 or abs(value) < 1e-3):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell) if not isinstance(cell, str) else cell for cell in row) + " |")
    return lines


def discussion(path: Path) -> list[str]:
    return [path.read_text().rstrip(), ""] if path.exists() else []


def execution_table(execution: dict[str, dict], seeds: str) -> list[str]:
    """'Cells, seeds and cost' section of the fixed report template."""
    rows = [[name, seeds, fmt(entry["sim_seconds"], 0), fmt(entry["wall_seconds"], 0), fmt(entry["sim_seconds"] / entry["wall_seconds"] if entry["wall_seconds"] else None, 2)]
            for name, entry in sorted(execution.items())]
    total_sim = sum(entry["sim_seconds"] for entry in execution.values())
    total_wall = sum(entry["wall_seconds"] for entry in execution.values())
    rows.append(["total", "", fmt(total_sim, 0), fmt(total_wall, 0), fmt(total_sim / total_wall if total_wall else None, 2)])
    return ["## Cells, seeds and cost", "", "Wall time is summed over worker processes (16 in parallel on the host).", ""] + \
        table(["Cell", "seeds", "simulated [s]", "worker wall [s]", "sim-s per worker-s"], rows) + [""]


def artifacts(paths: list[str]) -> list[str]:
    return ["## Artifacts", ""] + [f"- `{path}`" for path in paths] + [""]


EXPLORATORY = ("*Exploratory continuation.* Phase 2 did not pass its gate (NOT-GO), so by plan III.3 this verdict carries no gate authority; "
               "the phase was executed at the user's instruction to run the plan to the end.")
