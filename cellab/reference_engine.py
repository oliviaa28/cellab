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


def _eval_count_condition(condition: Mapping[str, Any], neighbor_vals: list[int]) -> bool:
    """Evaluate a count-based condition on neighbor values."""
    target_state = condition["count"]
    cnt = neighbor_vals.count(target_state)

    if "min" in condition and cnt < condition["min"]:
        return False
    if "max" in condition and cnt > condition["max"]:
        return False
    if "in" in condition and cnt not in condition["in"]:
        return False
    return True


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
        return all(_eval_condition_cell(part, cell_val, neighbor_vals) for part in condition["all"])

    if "any" in condition:
        return any(_eval_condition_cell(part, cell_val, neighbor_vals) for part in condition["any"])

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
    if cell_val not in set(rule["from_codes"]):
        return False, None

    if not _eval_condition_cell(rule.get("when"), cell_val, neighbor_vals):
        return False, None

    prob = rule.get("probability", 1.0)
    if prob < 1.0 and prob_val is not None and prob_val >= prob:
        return False, None

    return True, rule.get("to_code")


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
    outside = int(world["outside_code"]) if boundary == "fixed" else None
    offsets = automaton["neighbor_offsets"]

    rows, cols = cells.shape
    result_cells = cells.copy()
    unresolved = np.ones((rows, cols), dtype=bool)

    for rule in automaton["compiled_rules"]:
        prob = rule.get("probability", 1.0)
        rand_matrix = rng.random((rows, cols)) if prob < 1.0 else None

        for r in range(rows):
            for c in range(cols):
                if not unresolved[r, c]:
                    continue

                cell_val = int(cells[r, c])
                neighbor_vals = [
                    _get_neighbor_val(cells, r, c, dr, dc, boundary, outside)
                    for dr, dc in offsets
                ]
                prob_val = rand_matrix[r, c] if rand_matrix is not None else None

                matched, to_code = _apply_rule_to_cell(rule, cell_val, neighbor_vals, prob_val)
                if matched:
                    if to_code is not None:
                        result_cells[r, c] = to_code
                    unresolved[r, c] = False

    new_state = dict(state)
    new_state["cells"] = result_cells
    new_state["values"] = {name: val.copy() for name, val in state.get("values", {}).items()}
    new_state["agents"] = [dict(agent) for agent in state.get("agents", [])]
    new_state["generation"] = state.get("generation", 0) + 1
    return new_state