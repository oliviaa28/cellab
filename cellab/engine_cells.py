"""Vectorized cellular automaton stepper.

The engine works only on the *loaded* automaton described in docs/contract.md:

- ``compiled_rules``: rules with state names already translated into codes
  (``from_codes``, ``to_code``) and parameters already replaced by numbers;
- ``neighbor_offsets``: neighbor offsets in the fixed contract order;
- ``world``: ``boundary`` ("fixed" or "toroidal"), ``outside_code`` and
  ``outside_values``.

The loader guarantees these keys exist and are valid, so the engine does not
re-validate the format. It only checks the things that come from the caller
(the state and the random generator).

Random numbers (contract, decision 5): at every step, for every rule whose
probability is smaller than 1, one full matrix of random numbers is drawn,
in rule order, even if no cell uses it. The reference engine must follow
exactly the same rule.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _boundary(automaton: Mapping[str, Any]) -> tuple[str, int | None]:
    """Read the boundary from automaton["world"]; there is no silent default."""
    if "world" not in automaton:
        raise KeyError("loaded automaton has no 'world'; it must be loaded together with an experiment")
    world = automaton["world"]
    boundary = world["boundary"]
    if boundary == "fixed":
        return boundary, int(world["outside_code"])
    if boundary == "toroidal":
        return boundary, None
    raise ValueError(f"unsupported boundary type: {boundary!r}")


def _shift(cells: np.ndarray, offset: tuple[int, int], boundary: str, outside: int | None) -> np.ndarray:
    """Return a matrix where position (r, c) holds the neighbor of (r, c) in direction offset."""
    dr, dc = offset
    if boundary == "toroidal":
        return np.roll(cells, shift=(-dr, -dc), axis=(0, 1))

    rows, cols = cells.shape
    shifted = np.full(cells.shape, outside, dtype=cells.dtype)
    dst_r0, dst_r1 = max(0, -dr), min(rows, rows - dr)
    dst_c0, dst_c1 = max(0, -dc), min(cols, cols - dc)
    src_r0, src_r1 = max(0, dr), min(rows, rows + dr)
    src_c0, src_c1 = max(0, dc), min(cols, cols + dc)
    shifted[dst_r0:dst_r1, dst_c0:dst_c1] = cells[src_r0:src_r1, src_c0:src_c1]
    return shifted


def _count_mask(condition: Mapping[str, Any], cells: np.ndarray, neighbors: list[np.ndarray]) -> np.ndarray:
    count = np.zeros(cells.shape, dtype=np.int64)
    for neighbor in neighbors:
        count += neighbor == condition["count"]

    mask = np.ones(cells.shape, dtype=bool)
    if "min" in condition:
        mask &= count >= condition["min"]
    if "max" in condition:
        mask &= count <= condition["max"]
    if "in" in condition:
        mask &= np.isin(count, condition["in"])
    return mask


def _condition_mask(condition: Mapping[str, Any] | None, cells: np.ndarray, neighbors: list[np.ndarray]) -> np.ndarray:
    if condition is None:
        return np.ones(cells.shape, dtype=bool)

    if "count" in condition:
        return _count_mask(condition, cells, neighbors)

    if "pattern" in condition:
        # 1D neighbors come in contract order [W, E]; the center is the cell itself.
        left, center, right = condition["pattern"]
        return (neighbors[0] == left) & (cells == center) & (neighbors[1] == right)

    if "all" in condition:
        result = np.ones(cells.shape, dtype=bool)
        for part in condition["all"]:
            result &= _condition_mask(part, cells, neighbors)
        return result

    if "any" in condition:
        result = np.zeros(cells.shape, dtype=bool)
        for part in condition["any"]:
            result |= _condition_mask(part, cells, neighbors)
        return result

    if "not" in condition:
        return ~_condition_mask(condition["not"], cells, neighbors)

    if "expr" in condition:
        raise NotImplementedError("expr conditions are part of Phase 2")

    raise ValueError(f"unknown condition: {condition!r}")


def step_cells(
    automaton: Mapping[str, Any],
    state: Mapping[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Apply one synchronous generation and return a new state.

    The input state is never modified.
    """
    cells = np.asarray(state["cells"])
    if cells.ndim != 2 or not np.issubdtype(cells.dtype, np.integer):
        raise TypeError("state['cells'] must be a two-dimensional integer NumPy array")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    boundary, outside = _boundary(automaton)
    neighbors = [_shift(cells, offset, boundary, outside) for offset in automaton["neighbor_offsets"]]

    result_cells = cells.copy()
    unresolved = np.ones(cells.shape, dtype=bool)  # cells not yet changed by an earlier rule

    for rule in automaton["compiled_rules"]:
        matches = unresolved & np.isin(cells, rule["from_codes"])
        matches &= _condition_mask(rule["when"], cells, neighbors)

        if rule["probability"] < 1:
            matches &= rng.random(cells.shape) < rule["probability"]

        if rule["to_code"] is not None:
            result_cells[matches] = rule["to_code"]
        unresolved[matches] = False

    new_state = dict(state)
    new_state["cells"] = result_cells
    new_state["values"] = {name: value.copy() for name, value in state.get("values", {}).items()}
    new_state["agents"] = [dict(agent) for agent in state.get("agents", [])]
    new_state["generation"] = state.get("generation", 0) + 1
    return new_state
