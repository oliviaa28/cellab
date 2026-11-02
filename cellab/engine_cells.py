"""Vectorized cellular automaton stepper (format 2.0, Phase 1).

The loader is expected to translate state names in ``from``, ``to``,
``count`` and ``pattern`` into integer state codes. Parameters in the
automaton are expected to be resolved with experiment overrides before this
function is called.
"""


from __future__ import annotations # for python 3.9 compatibility with type hints

from collections.abc import Mapping
from typing import Any

import numpy as np

# compute the offsets for a given neighborhood type and radius; Phase 1 supports radius 1 only
def _neighborhood_offsets(neighborhood: Mapping[str, Any]) -> tuple[str, tuple[tuple[int, int], ...]]:
    kind = neighborhood.get("type")
    radius = neighborhood.get("radius", 1)
    if radius != 1:
        raise ValueError("step_cells supports neighborhood radius 1 only")
    if kind == "1d":
        return kind, ((0, -1), (0, 1))
    if kind == "von_neumann":
        return kind, ((-1, 0), (1, 0), (0, -1), (0, 1))
    if kind == "moore":
        return kind, tuple(
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
        )
    raise ValueError(f"unsupported neighborhood type: {kind!r}")


def _boundary_config(automaton: Mapping[str, Any]) -> tuple[str, int | None]:
    """Read experiment boundary data attached by the loader, if present."""

    world = automaton.get("world", {})
    boundary = world.get("boundary", automaton.get("boundary", "toroidal"))

    if boundary not in ("fixed", "toroidal"):
        raise ValueError(f"unsupported boundary type: {boundary!r}")
    
    outside = world.get("outside_state", automaton.get("outside_state"))

    if boundary == "fixed" and outside is None:
        raise ValueError("fixed boundaries require an integer outside_state")
    
    if outside is not None and not isinstance(outside, (int, np.integer)):
        raise TypeError("outside_state must be an integer state code")
    
    return boundary, None if outside is None else int(outside)


def _shift(cells: np.ndarray, offset: tuple[int, int], boundary: str, outside: int | None) -> np.ndarray:
    dr, dc = offset
    if boundary == "toroidal":
        return np.roll(cells, shift=(-dr, -dc), axis=(0, 1))
    
    rows, cols = cells.shape

    shifted = np.full(cells.shape, outside, dtype=np.int64)

    dst_r0, dst_r1 = max(0, -dr), min(rows, rows - dr)
    dst_c0, dst_c1 = max(0, -dc), min(cols, cols - dc)
    src_r0, src_r1 = max(0, dr), min(rows, rows + dr)
    src_c0, src_c1 = max(0, dc), min(cols, cols + dc)
    shifted[dst_r0:dst_r1, dst_c0:dst_c1] = cells[src_r0:src_r1, src_c0:src_c1]
    return shifted


def _parameter(value: Any, parameters: Mapping[str, Any], *, field: str) -> Any:
    if isinstance(value, str):
        if value not in parameters:
            raise ValueError(f"unknown {field} parameter {value!r}")
        return parameters[value]
    return value


def _condition_mask(
    condition: Mapping[str, Any] | None,
    cells: np.ndarray,
    neighbors: tuple[np.ndarray, ...],
    offsets: tuple[tuple[int, int], ...],
    neighborhood_type: str,
    parameters: Mapping[str, Any],
) -> np.ndarray:
    if not condition:
        return np.ones(cells.shape, dtype=bool)
    if not isinstance(condition, Mapping):
        raise ValueError("each when condition must be an object")
    operators = {"count", "pattern", "all", "any", "not", "expr"}
    present_operators = operators.intersection(condition)
    if len(present_operators) != 1:
        raise ValueError("each when condition must contain exactly one condition operator")

    if "count" in condition:
        target = condition["count"]
        count = np.zeros(cells.shape, dtype=np.int64)
        for neighbor in neighbors:
            count += neighbor == target
        if "min" in condition:
            count_min = _parameter(condition["min"], parameters, field="min")
            if count_min > len(offsets):
                return np.zeros(cells.shape, dtype=bool)
            count_mask = count >= count_min
        elif "max" in condition:
            count_mask = count <= _parameter(condition["max"], parameters, field="max")
        elif "in" in condition:
            values = [_parameter(value, parameters, field="in") for value in condition["in"]]
            count_mask = np.isin(count, values)
        else:
            raise ValueError("count condition requires min, max or in")
        if "max" in condition and "min" in condition:
            count_max = _parameter(condition["max"], parameters, field="max")
            if count_max < 0:
                return np.zeros(cells.shape, dtype=bool)
            count_mask &= count <= count_max
        return count_mask

    if "pattern" in condition:
        if neighborhood_type != "1d" or cells.shape[0] != 1:
            raise ValueError("pattern conditions are supported only on 1D grids")
        pattern = condition["pattern"]
        if len(pattern) != 3:
            raise ValueError("1D pattern must contain left, center and right states")
        # The two 1D neighbors are ordered left then right; center is the cell.
        return (neighbors[0] == pattern[0]) & (cells == pattern[1]) & (neighbors[1] == pattern[2])

    if "all" in condition:
        parts = condition["all"]
        if not parts:
            return np.ones(cells.shape, dtype=bool)
        result = np.ones(cells.shape, dtype=bool)
        for part in parts:
            result &= _condition_mask(part, cells, neighbors, offsets, neighborhood_type, parameters)
        return result
    if "any" in condition:
        parts = condition["any"]
        result = np.zeros(cells.shape, dtype=bool)
        for part in parts:
            result |= _condition_mask(part, cells, neighbors, offsets, neighborhood_type, parameters)
        return result
    if "not" in condition:
        return ~_condition_mask(condition["not"], cells, neighbors, offsets, neighborhood_type, parameters)
    if "expr" in condition:
        raise NotImplementedError("expr conditions are part of Phase 2")
    raise ValueError(f"unsupported condition: {next(iter(condition))!r}")


def step_cells(
    automaton: Mapping[str, Any],
    state: Mapping[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Apply one synchronous cellular generation and return a fresh state.

    Expected state keys are ``cells`` (integer NumPy matrix), ``values``,
    ``agents`` and ``generation``. Phase 1 leaves the latter three unchanged;
    arrays are copied so neither the input state nor its cell matrix is mutated.
    """
    cells = np.asarray(state["cells"])

    #if the input state is not a 2D integer array, raise an error
    if cells.ndim != 2 or not np.issubdtype(cells.dtype, np.integer):
        raise TypeError("state['cells'] must be a two-dimensional integer NumPy array")
    
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    
    neighborhood_type, offsets = _neighborhood_offsets(automaton["neighborhood"])

    boundary, outside = _boundary_config(automaton)

    neighbors = tuple(_shift(cells, offset, boundary, outside) for offset in offsets)
    parameters = automaton.get("parameters", {})
    result_cells = cells.copy()
    unresolved = np.ones(cells.shape, dtype=bool) # True for cells that have not yet been updated by a rule

    for rule in automaton["rules"]:
        source = rule["from"]
        if source == "*":
            source_mask = np.ones(cells.shape, dtype=bool)
        elif isinstance(source, (list, tuple, np.ndarray)):
            source_mask = np.isin(cells, source) 
        else:
            source_mask = cells == source
        matches = unresolved & source_mask
        if "when" in rule:
            matches &= _condition_mask(
                rule["when"], cells, neighbors, offsets, neighborhood_type, parameters
            )

        if "probability" in rule:
            probability = _parameter(rule["probability"], parameters, field="probability")
            if not isinstance(probability, (int, float, np.integer, np.floating)):
                raise TypeError("rule probability must resolve to a number")
            if not 0 <= probability <= 1:
                raise ValueError("rule probability must be between 0 and 1")
            # Draw once for every explicitly probabilistic rule, in rule order.
            matches &= rng.random(cells.shape) < probability

        if "to" in rule:
            result_cells[matches] = rule["to"]
        unresolved[matches] = False

    new_state = dict(state)
    new_state["cells"] = result_cells
    new_state["values"] = {name: value.copy() for name, value in state.get("values", {}).items()}
    new_state["agents"] = [dict(agent) for agent in state.get("agents", [])]
    new_state["generation"] = state.get("generation", 0) + 1
    return new_state
