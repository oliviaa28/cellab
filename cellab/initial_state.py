"""initial_state.py – Create the initial state from an experiment description.

Phase 1: supports cells (fill, random, manual, pattern) and values
(constant with optional points, random, manual). Agents are stubbed
(returns [] for Phase 1).

Every random operation uses the single ``rng`` passed in – never
``np.random`` or Python's ``random``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

def _cells_fill(spec: dict, automaton: dict, rows: int, cols: int,
                rng: np.random.Generator) -> np.ndarray:
    """All cells set to one state."""
    code = automaton["state_codes"][spec["state"]]
    return np.full((rows, cols), code, dtype=np.int64)


def _cells_random(spec: dict, automaton: dict, rows: int, cols: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Random cells according to proportions that sum to 1."""
    proportions = spec["proportions"]
    codes = [automaton["state_codes"][s] for s in proportions.keys()]
    probs = [float(p) for p in proportions.values()]

    # Normalize probabilities slightly to guard against floating point inaccuracies
    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]

    flat = rng.choice(codes, size=rows * cols, p=probs)
    return flat.reshape((rows, cols)).astype(np.int64)


def _cells_manual(spec: dict, automaton: dict, rows: int, cols: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Grid written with symbol characters, one string per row."""
    grid_strings = spec["grid"]
    symbols = automaton["symbols"]  # symbol -> code
    cells = np.zeros((rows, cols), dtype=np.int64)
    for r, row_str in enumerate(grid_strings):
        for c, ch in enumerate(row_str):
            cells[r, c] = symbols[ch]
    return cells


def _cells_pattern(spec: dict, automaton: dict, rows: int, cols: int,
                   rng: np.random.Generator) -> np.ndarray:
    """A small pattern placed on a background at a given position."""
    bg_code = automaton["state_codes"][spec["background"]]
    cells = np.full((rows, cols), bg_code, dtype=np.int64)

    pattern_rows = spec["pattern"]          # list of symbol-strings
    pr, pc = spec["position"]               # [row, col]
    symbols = automaton["symbols"]

    for dr, row_str in enumerate(pattern_rows):
        for dc, ch in enumerate(row_str):
            r, c = pr + dr, pc + dc
            if 0 <= r < rows and 0 <= c < cols:
                cells[r, c] = symbols[ch]
    return cells


_CELLS_MODES = {
    "fill":    _cells_fill,
    "random":  _cells_random,
    "manual":  _cells_manual,
    "pattern": _cells_pattern,
}


# ---------------------------------------------------------------------------
# Values (numeric variables attached to cells)
# ---------------------------------------------------------------------------

def _values_constant(var_spec: dict, var_name: str, automaton: dict,
                     rows: int, cols: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Constant value everywhere, with optional points overridden."""
    vtype = automaton.get("variable_types", {}).get(var_name, "float")
    dtype = np.int64 if vtype == "int" else np.float64
    default_val = automaton.get("variable_defaults", {}).get(var_name, 0)
    base_val = var_spec.get("value", default_val)
    grid = np.full((rows, cols), base_val, dtype=dtype)

    # Each point has position [r, c] (schema 2.0) or row/col
    for pt in var_spec.get("points", []):
        if "position" in pt:
            r, c = pt["position"]
        else:
            r, c = pt["row"], pt["col"]
        if 0 <= r < rows and 0 <= c < cols:
            grid[r, c] = pt["value"]
    return grid


def _values_random(var_spec: dict, var_name: str, automaton: dict,
                   rows: int, cols: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Random values in [min, max]."""
    vtype = automaton.get("variable_types", {}).get(var_name, "float")
    lo, hi = var_spec["min"], var_spec["max"]
    if vtype == "int":
        return rng.integers(int(lo), int(hi), size=(rows, cols), endpoint=True).astype(np.int64)
    else:
        return rng.uniform(lo, hi, size=(rows, cols)).astype(np.float64)


def _values_manual(var_spec: dict, var_name: str, automaton: dict,
                   rows: int, cols: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Grid of numbers provided directly."""
    vtype = automaton.get("variable_types", {}).get(var_name, "float")
    dtype = np.int64 if vtype == "int" else np.float64
    return np.array(var_spec["grid"], dtype=dtype)


_VALUES_MODES = {
    "constant": _values_constant,
    "random":   _values_random,
    "manual":   _values_manual,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_initial_state(
    automaton: dict[str, Any],
    experiment: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Build the generation-0 state from an automaton and experiment.

    Returns a state dict: ``{cells, values, agents, generation}``.
    """
    # World dimensions come from experiment["world"] per contract 6.6
    world_spec = experiment.get("world") or automaton.get("world") or {}
    rows = int(world_spec["rows"])
    cols = int(world_spec["cols"])

    # --- cells -----------------------------------------------------------
    cell_spec = experiment["initial_state"]["cells"]
    mode = cell_spec["mode"]
    cells = _CELLS_MODES[mode](cell_spec, automaton, rows, cols, rng)

    # --- values ----------------------------------------------------------
    values: dict[str, np.ndarray] = {}
    val_specs = experiment["initial_state"].get("values", {})
    var_types = automaton.get("variable_types", {})

    for var_name in var_types:
        if var_name in val_specs:
            vspec = val_specs[var_name]
            vmode = vspec["mode"]
            values[var_name] = _VALUES_MODES[vmode](
                vspec, var_name, automaton, rows, cols, rng)
        else:
            # Use the default value everywhere
            vtype = var_types[var_name]
            dtype = np.int64 if vtype == "int" else np.float64
            default = automaton.get("variable_defaults", {}).get(var_name, 0)
            values[var_name] = np.full((rows, cols), default, dtype=dtype)

    # --- agents (stub for Phase 1) ---------------------------------------
    agents: list[dict] = []

    return {
        "cells": cells,
        "values": values,
        "agents": agents,
        "generation": 0,
    }
