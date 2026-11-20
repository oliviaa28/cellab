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

Random numbers: at every step, in rule order, one full matrix of random
numbers is drawn for every rule whose probability is smaller than 1 and for
every random() in its formulas, even if no cell uses it. Inside a rule the
order is: the random() calls of its condition, its probability, the random()
calls of its set formulas. The reference engine follows exactly the same order.

Formulas (expr conditions and set) are computed by cellab.expressions on the
whole grid at once. The engine prepares what they need (the context): the
neighbors' states and variables, already shifted, with the boundary applied.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from cellab.expressions import ExpressionRuntimeError, evaluate_expression, parse_expression


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


def _shift(cells: np.ndarray, offset: tuple[int, int], boundary: str, outside) -> np.ndarray:
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


def _neighbor_layers(cells, values, automaton, boundary, outside) -> list[dict[str, np.ndarray]]:
    """One dict per direction: the neighbor's state ("cells") and the neighbor's variables.

    At a fixed boundary, variables outside the grid have the world's outside values.
    """
    outside_values = automaton["world"].get("outside_values", {})
    layers = []
    for offset in automaton["neighbor_offsets"]:
        layer = {"cells": _shift(cells, offset, boundary, outside)}
        for name, matrix in values.items():
            layer[name] = _shift(matrix, offset, boundary, outside_values.get(name, 0))
        layers.append(layer)
    return layers


def _formula(text: str, automaton: Mapping[str, Any], expected: str) -> dict:
    """The formula, ready to compute (parse_expression remembers formulas it has already read)."""
    return parse_expression(
        text,
        automaton.get("state_names", []),
        list(automaton.get("variable_types", {})),
        list(automaton.get("parameters", {})),
        expected,
    )


def _count_mask(condition: Mapping[str, Any], cells: np.ndarray, neighbors: list[np.ndarray]) -> np.ndarray:
    # "count" is one state code or a list of codes (neighbors in any of them are counted)
    targets = condition["count"]
    if not isinstance(targets, list):
        targets = [targets]

    count = np.zeros(cells.shape, dtype=np.int64)
    for neighbor in neighbors:
        count += np.isin(neighbor, targets)

    mask = np.ones(cells.shape, dtype=bool)
    if "min" in condition:
        mask &= count >= condition["min"]
    if "max" in condition:
        mask &= count <= condition["max"]
    if "in" in condition:
        mask &= np.isin(count, condition["in"])
    return mask


def _condition_mask(condition: Mapping[str, Any] | None, grid: Mapping[str, Any], where: np.ndarray) -> np.ndarray:
    """Where the condition is true.

    where: the cells for which the answer matters. As in the reference engine, a later part of
    all / any only matters where the earlier parts did not decide. This changes nothing for the
    result; it only means that a division by 0 where the result is not used is not an error.
    """
    cells = grid["cells"]
    if condition is None:
        return np.ones(cells.shape, dtype=bool)

    if "count" in condition:
        return _count_mask(condition, cells, grid["neighbors"])

    if "pattern" in condition:
        # 1D neighbors come in contract order [W, E]; the center is the cell itself.
        left, center, right = condition["pattern"]
        return (grid["neighbors"][0] == left) & (cells == center) & (grid["neighbors"][1] == right)

    if "all" in condition:
        result = np.ones(cells.shape, dtype=bool)
        for part in condition["all"]:
            result &= _condition_mask(part, grid, where & result)
        return result

    if "any" in condition:
        result = np.zeros(cells.shape, dtype=bool)
        for part in condition["any"]:
            result |= _condition_mask(part, grid, where & ~result)
        return result

    if "not" in condition:
        return ~_condition_mask(condition["not"], grid, where)

    if "expr" in condition:
        formula = _formula(condition["expr"], grid["automaton"], "bool")
        return evaluate_expression(formula, grid["context"], mask=where)

    raise ValueError(f"unknown condition: {condition!r}")


def _apply_set(rule: Mapping[str, Any], grid: Mapping[str, Any], matches: np.ndarray,
               new_values: dict[str, np.ndarray]) -> None:
    """Compute the set formulas from the previous generation; write them where the rule applies."""
    for name, text in rule["set"].items():
        formula = _formula(text, grid["automaton"], "number")
        value = evaluate_expression(formula, grid["context"], mask=matches)
        target = new_values[name]
        if np.issubdtype(target.dtype, np.integer):
            value = np.floor(value)    # an int variable is rounded down
        target[matches] = value[matches].astype(target.dtype)


def _make_grid(automaton: Mapping[str, Any], cells: np.ndarray, values: dict, rng) -> dict[str, Any]:
    """Everything one step needs, computed once: neighbors and the context of the formulas."""
    boundary, outside = _boundary(automaton)
    layers = _neighbor_layers(cells, values, automaton, boundary, outside)
    return {
        "automaton": automaton,
        "cells": cells,
        "neighbors": [layer["cells"] for layer in layers],
        "context": {
            "self": values,
            "neighbors": layers,
            "state_codes": automaton.get("state_codes", {}),
            "parameters": automaton.get("parameters", {}),
            "rng": rng,
            "shape": cells.shape,
        },
    }


def step_cells(
    automaton: Mapping[str, Any],
    state: Mapping[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Apply one synchronous generation and return a new state.

    The input state is never modified.
    A formula that fails (for example a division by 0) raises ExpressionRuntimeError,
    with the rule, the formula and the generation in the message.
    """
    cells = np.asarray(state["cells"])
    if cells.ndim != 2 or not np.issubdtype(cells.dtype, np.integer):
        raise TypeError("state['cells'] must be a two-dimensional integer NumPy array")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    values = {name: np.asarray(matrix) for name, matrix in state.get("values", {}).items()}
    grid = _make_grid(automaton, cells, values, rng)
    generation = state.get("generation", 0)

    result_cells = cells.copy()
    new_values = {name: matrix.copy() for name, matrix in values.items()}
    unresolved = np.ones(cells.shape, dtype=bool)  # cells not yet changed by an earlier rule

    for index, rule in enumerate(automaton["compiled_rules"]):
        try:
            candidates = unresolved & np.isin(cells, rule["from_codes"])
            matches = candidates & _condition_mask(rule["when"], grid, candidates)

            if rule["probability"] < 1:
                matches &= rng.random(cells.shape) < rule["probability"]

            if rule.get("set"):
                _apply_set(rule, grid, matches, new_values)
        except ExpressionRuntimeError as error:
            raise ExpressionRuntimeError(f"rule {index}, generation {generation + 1}: {error}") from None

        if rule["to_code"] is not None:
            result_cells[matches] = rule["to_code"]
        unresolved[matches] = False

    new_state = dict(state)
    new_state["cells"] = result_cells
    new_state["values"] = new_values
    new_state["agents"] = [dict(agent) for agent in state.get("agents", [])]
    new_state["generation"] = generation + 1
    return new_state
