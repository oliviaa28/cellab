"""Reference cellular automaton stepper.

A slow, simple, cell-by-cell implementation of the cellular engine semantics
used as the ground truth for differential testing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _get_neighbor_val(
    cells: np.ndarray,
    r: int,
    c: int,
    dr: int,
    dc: int,
    boundary: str,
    outside: int | None,
) -> int:
    nr = r + dr
    nc = c + dc
    rows, cols = cells.shape

    if boundary == "toroidal":
        return int(cells[nr % rows, nc % cols])

    if 0 <= nr < rows and 0 <= nc < cols:
        return int(cells[nr, nc])

    return int(outside)


def _neighbor_values(grid: Mapping[str, Any], r: int, c: int) -> list[int]:
    """The neighbor states of cell (r, c), in the order of the neighbor offsets."""
    values = []
    for dr, dc in grid["offsets"]:
        values.append(_get_neighbor_val(grid["cells"], r, c, dr, dc,
                                        grid["boundary"], grid["outside"]))
    return values


def _eval_count_condition(condition: Mapping[str, Any], neighbor_vals: list[int]) -> bool:
    """Evaluate a count-based condition on neighbor values."""
    targets = condition["count"]
    if not isinstance(targets, list):
        targets = [targets]

    cnt = 0
    for value in neighbor_vals:
        if value in targets:
            cnt += 1

    if "min" in condition and cnt < condition["min"]:
        return False
    if "max" in condition and cnt > condition["max"]:
        return False
    if "in" in condition and cnt not in condition["in"]:
        return False
    return True


def _eval_all(parts: list, cell_val: int, neighbor_vals: list[int]) -> bool:
    """True if every part is true."""
    for part in parts:
        if not _eval_condition_cell(part, cell_val, neighbor_vals):
            return False
    return True


def _eval_any(parts: list, cell_val: int, neighbor_vals: list[int]) -> bool:
    """True if at least one part is true."""
    for part in parts:
        if _eval_condition_cell(part, cell_val, neighbor_vals):
            return True
    return False


def _eval_condition_cell(
    condition: Mapping[str, Any] | None,
    cell_val: int,
    neighbor_vals: list[int],
) -> bool:
    if condition is None:
        return True

    if "count" in condition:
        return _eval_count_condition(condition, neighbor_vals)

    if "pattern" in condition:
        left, center, right = condition["pattern"]
        return neighbor_vals[0] == left and cell_val == center and neighbor_vals[1] == right

    if "all" in condition:
        return _eval_all(condition["all"], cell_val, neighbor_vals)

    if "any" in condition:
        return _eval_any(condition["any"], cell_val, neighbor_vals)

    if "not" in condition:
        return not _eval_condition_cell(condition["not"], cell_val, neighbor_vals)

    if "expr" in condition:
        raise NotImplementedError("expr conditions belong to Phase 2")

    raise ValueError(f"Unknown condition: {condition!r}")


def _apply_rule_to_cell(
    rule: Mapping[str, Any],
    cell_val: int,
    neighbor_vals: list[int],
    prob_val: float | None,
) -> tuple[bool, int | None]:
    """Check rule match. Returns (matches, to_code)."""
    if cell_val not in rule["from_codes"]:
        return False, None

    if not _eval_condition_cell(rule.get("when"), cell_val, neighbor_vals):
        return False, None

    prob = rule.get("probability", 1.0)
    if prob < 1.0 and prob_val is not None and prob_val >= prob:
        return False, None

    return True, rule.get("to_code")


def _process_single_cell(
    rule: Mapping[str, Any],
    grid: dict[str, Any],
    coord: tuple[int, int],
    rand_matrix: np.ndarray | None,
) -> None:
    """Apply one rule to one cell, if the cell is not resolved yet."""
    r, c = coord
    if not grid["unresolved"][r, c]:
        return

    cell_val = int(grid["cells"][r, c])
    neighbor_vals = _neighbor_values(grid, r, c)
    prob_val = rand_matrix[r, c] if rand_matrix is not None else None

    matched, to_code = _apply_rule_to_cell(rule, cell_val, neighbor_vals, prob_val)
    if not matched:
        return

    if to_code is not None:
        grid["result_cells"][r, c] = to_code
    grid["unresolved"][r, c] = False


def _apply_rule_to_grid(
    rule: Mapping[str, Any],
    grid: dict[str, Any],
    rand_matrix: np.ndarray | None,
) -> None:
    """Apply a single rule to all unresolved cells in the grid."""
    for coord in np.ndindex(grid["cells"].shape):
        _process_single_cell(rule, grid, coord, rand_matrix)


def step_cells_reference(
    automaton: Mapping[str, Any],
    state: Mapping[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Execute one synchronous step cell-by-cell and return a new state."""
    cells = np.asarray(state["cells"])
    if cells.ndim != 2 or not np.issubdtype(cells.dtype, np.integer):
        raise TypeError("state['cells'] must be a 2D integer NumPy array")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    world = automaton["world"]
    boundary = world["boundary"]

    # Everything one step needs to know about the grid
    grid = {
        "cells": cells,
        "result_cells": cells.copy(),
        "unresolved": np.ones(cells.shape, dtype=bool),
        "offsets": automaton["neighbor_offsets"],
        "boundary": boundary,
        "outside": int(world["outside_code"]) if boundary == "fixed" else None,
    }

    for rule in automaton["compiled_rules"]:
        if rule.get("set"):
            raise NotImplementedError("set formulas belong to Phase 2")

        prob = rule.get("probability", 1.0)
        rand_matrix = rng.random(cells.shape) if prob < 1.0 else None
        _apply_rule_to_grid(rule, grid, rand_matrix)

    values = {}
    for name, val in state.get("values", {}).items():
        values[name] = val.copy()
    agents = []
    for agent in state.get("agents", []):
        agents.append(dict(agent))

    new_state = dict(state)
    new_state["cells"] = grid["result_cells"]
    new_state["values"] = values
    new_state["agents"] = agents
    new_state["generation"] = state.get("generation", 0) + 1
    return new_state
