"""Random cellular automata generator.

Produces random automata and experiments as raw JSON (format 2.0), exactly
like the files in automata/ and experiments/. Every generated case goes
through the validator and the loader, so the result has the same shape as
any real automaton and can be used by every module:
initial_state, simulation, storage, engines, dataset building.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cellab.loader import load_automaton, load_experiment
from cellab.validator import has_errors, validate_automaton, validate_experiment


NEIGHBOR_COUNTS = {"moore": 8, "von_neumann": 4, "1d": 2}

SYMBOLS = "0123456789"

COLORS = ["#1E1E1E", "#F2C14E", "#3A7BD5", "#D1495B", "#66A182",
          "#8E6C8A", "#EDAE49", "#00798C", "#30638E", "#F4F1DE"]

MAX_STATES = len(SYMBOLS)


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

def _random_count_condition(names: list[str], max_neighbors: int,
                            rng: np.random.Generator) -> dict[str, Any]:
    """A count condition: {"count": state, "min": a, "max": b} or {"count": state, "in": [...]}."""
    target = names[int(rng.integers(0, len(names)))]

    if rng.random() < 0.5:
        low = int(rng.integers(0, max_neighbors + 1))
        high = int(rng.integers(low, max_neighbors + 1))
        return {"count": target, "min": low, "max": high}

    k = int(rng.integers(1, min(3, max_neighbors + 1)))
    chosen = rng.choice(max_neighbors + 1, size=k, replace=False)
    values = []
    for value in sorted(chosen):
        values.append(int(value))
    return {"count": target, "in": values}


def _random_pattern_condition(names: list[str], rng: np.random.Generator) -> dict[str, Any]:
    """A 1D pattern condition: {"pattern": [W, cell, E]}."""
    pattern = []
    for _ in range(3):
        pattern.append(names[int(rng.integers(0, len(names)))])
    return {"pattern": pattern}


def _random_combined_condition(kind: str, neighborhood_type: str, names: list[str],
                               rng: np.random.Generator, depth: int) -> dict[str, Any] | None:
    """An all / any / not condition built from smaller random conditions."""
    if kind == "not":
        part = _random_condition(neighborhood_type, names, rng, depth + 1)
        if part is None:
            return None
        return {"not": part}

    parts = []
    for _ in range(2):
        part = _random_condition(neighborhood_type, names, rng, depth + 1)
        if part is not None:
            parts.append(part)
    if not parts:
        return None
    return {kind: parts}


def _random_condition(neighborhood_type: str, names: list[str],
                      rng: np.random.Generator, depth: int = 0) -> dict[str, Any] | None:
    """A random condition, or None (rule without "when")."""
    if depth > 1 or rng.random() < 0.2:
        return None

    if neighborhood_type == "1d":
        kinds = ["count", "pattern"]
    else:
        kinds = ["count", "all", "any", "not"]
    kind = str(rng.choice(kinds))

    if kind == "count":
        return _random_count_condition(names, NEIGHBOR_COUNTS[neighborhood_type], rng)
    if kind == "pattern":
        return _random_pattern_condition(names, rng)
    return _random_combined_condition(kind, neighborhood_type, names, rng, depth)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def _random_from(names: list[str], rng: np.random.Generator) -> str | list[str]:
    """The "from" field: "*", one state name, or a list of state names."""
    if rng.random() < 0.2:
        return "*"

    k = int(rng.integers(1, len(names) + 1))
    chosen = rng.choice(len(names), size=k, replace=False)
    if k == 1:
        return names[int(chosen[0])]

    result = []
    for index in chosen:
        result.append(names[int(index)])
    return result


def _random_rule(neighborhood_type: str, names: list[str],
                 rng: np.random.Generator) -> dict[str, Any]:
    """One random rule in format 2.0 (state names, not codes)."""
    rule: dict[str, Any] = {
        "from": _random_from(names, rng),
        "to": names[int(rng.integers(0, len(names)))],
    }

    when = _random_condition(neighborhood_type, names, rng)
    if when is not None:
        rule["when"] = when

    if rng.random() >= 0.6:
        rule["probability"] = float(np.round(rng.uniform(0.1, 0.9), 2))
    return rule


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_random_automaton(
    rng: np.random.Generator,
    neighborhood_type: str | None = None,
    num_states: int | None = None,
    num_rules: int | None = None,
) -> dict[str, Any]:
    """Generate a random automaton as a raw JSON dict (format 2.0)."""
    if neighborhood_type is None:
        neighborhood_type = str(rng.choice(["moore", "von_neumann", "1d"]))
    if num_states is None:
        num_states = int(rng.integers(2, 6))
    if num_rules is None:
        num_rules = int(rng.integers(1, 8))
    if not 2 <= num_states <= MAX_STATES:
        raise ValueError(f"num_states must be between 2 and {MAX_STATES}, got {num_states}")

    states = []
    names = []
    for i in range(num_states):
        states.append({"name": f"S{i}", "symbol": SYMBOLS[i], "color": COLORS[i]})
        names.append(f"S{i}")

    rules = []
    for _ in range(num_rules):
        rules.append(_random_rule(neighborhood_type, names, rng))

    return {
        "format_version": "2.0",
        "kind": "cells",
        "name": f"Random_{neighborhood_type}_{num_states}states",
        "states": states,
        "neighborhood": {"type": neighborhood_type, "radius": 1},
        "rules": rules,
    }


def generate_random_experiment(
    raw_aut: dict[str, Any],
    rng: np.random.Generator,
    boundary: str | None = None,
    rows: int | None = None,
    cols: int | None = None,
    generations: int = 5,
) -> dict[str, Any]:
    """Generate a random experiment (raw JSON, format 2.0) for a raw automaton.

    1D automata always get a grid with one row.
    """
    if boundary is None:
        boundary = str(rng.choice(["fixed", "toroidal"]))
    if raw_aut["neighborhood"]["type"] == "1d":
        rows = 1
    elif rows is None:
        rows = int(rng.integers(4, 15))
    if cols is None:
        cols = int(rng.integers(5, 20))

    names = []
    for state in raw_aut["states"]:
        names.append(state["name"])

    world: dict[str, Any] = {"rows": rows, "cols": cols, "boundary": boundary}
    if boundary == "fixed":
        world["outside"] = {"state": names[int(rng.integers(0, len(names)))]}

    shares = rng.dirichlet(np.ones(len(names)))
    proportions = {}
    for name, share in zip(names, shares):
        proportions[name] = float(share)

    return {
        "format_version": "2.0",
        "automaton": "automata/random.json",
        "world": world,
        "initial_state": {"cells": {"mode": "random", "proportions": proportions}},
        "generations": generations,
        "storage": {"mode": "all"},
        "seed": int(rng.integers(0, 2**31 - 1)),
    }


def generate_random_case(
    rng: np.random.Generator,
    neighborhood_type: str | None = None,
    boundary: str | None = None,
) -> dict[str, Any]:
    """Generate a random automaton + experiment, validate both and load them.

    Returns ``{"raw_automaton", "raw_experiment", "automaton", "experiment"}``
    where automaton / experiment are loaded exactly like real files.
    Raises ValueError if the validator finds an error (a bug in this generator).
    """
    raw_aut = generate_random_automaton(rng, neighborhood_type)
    raw_exp = generate_random_experiment(raw_aut, rng, boundary)

    issues = validate_automaton(raw_aut)
    if not has_errors(issues):
        issues = issues + validate_experiment(raw_exp, raw_aut)
    if has_errors(issues):
        raise ValueError(f"Generated case is not valid: {issues}")

    automaton = load_automaton(raw_aut, raw_exp)
    experiment = load_experiment(raw_exp, automaton)
    return {
        "raw_automaton": raw_aut,
        "raw_experiment": raw_exp,
        "automaton": automaton,
        "experiment": experiment,
    }
