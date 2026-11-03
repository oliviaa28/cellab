"""Random cellular automata generator.

Produces structurally and semantically valid cellular automata for differential
testing and dataset building.
"""

from __future__ import annotations

import string
from typing import Any
import numpy as np


NEIGHBORHOOD_OFFSETS = {
    "moore": [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
    "von_neumann": [(-1, 0), (0, -1), (0, 1), (1, 0)],
    "1d": [(0, -1), (0, 1)],
}


def _random_condition(
    neighborhood_type: str,
    num_states: int,
    rng: np.random.Generator,
    depth: int = 0,
) -> dict[str, Any] | None:
    if depth > 1 or rng.random() < 0.2:
        return None

    cond_type = rng.choice(["count", "all", "any", "not"] if neighborhood_type != "1d" else ["count", "pattern"])
    max_neighbors = len(NEIGHBORHOOD_OFFSETS[neighborhood_type])

    if cond_type == "count":
        target = int(rng.integers(0, num_states))
        sub_type = rng.choice(["min_max", "in"])
        if sub_type == "min_max":
            low = int(rng.integers(0, max_neighbors + 1))
            high = int(rng.integers(low, max_neighbors + 1))
            return {"count": target, "min": low, "max": high}
        else:
            k = int(rng.integers(1, min(3, max_neighbors + 1)))
            vals = sorted(rng.choice(max_neighbors + 1, size=k, replace=False))
            return {"count": target, "in": [int(v) for v in vals]}

    if cond_type == "pattern" and neighborhood_type == "1d":
        return {"pattern": [int(rng.integers(0, num_states)) for _ in range(3)]}

    if cond_type == "not":
        sub = _random_condition(neighborhood_type, num_states, rng, depth + 1)
        return {"not": sub} if sub else None

    if cond_type in ("all", "any"):
        c1 = _random_condition(neighborhood_type, num_states, rng, depth + 1)
        c2 = _random_condition(neighborhood_type, num_states, rng, depth + 1)
        parts = [c for c in (c1, c2) if c is not None]
        return {cond_type: parts} if parts else None

    return None


def generate_random_automaton(
    rng: np.random.Generator,
    neighborhood_type: str | None = None,
    num_states: int | None = None,
    num_rules: int | None = None,
    boundary: str | None = None,
) -> dict[str, Any]:
    """Generate a fully compiled, valid random automaton ready for step_cells."""
    if neighborhood_type is None:
        neighborhood_type = str(rng.choice(["moore", "von_neumann", "1d"]))

    if num_states is None:
        num_states = int(rng.integers(2, 6))

    if num_rules is None:
        num_rules = int(rng.integers(1, 8))

    if boundary is None:
        boundary = str(rng.choice(["fixed", "toroidal"]))

    outside_code = int(rng.integers(0, num_states))
    offsets = NEIGHBORHOOD_OFFSETS[neighborhood_type]

    compiled_rules = []
    for _ in range(num_rules):
        # Choose from_codes
        if rng.random() < 0.2:
            from_codes = list(range(num_states))
        else:
            k = int(rng.integers(1, num_states + 1))
            from_codes = [int(x) for x in rng.choice(num_states, size=k, replace=False)]

        to_code = int(rng.integers(0, num_states))
        prob = 1.0 if rng.random() < 0.6 else float(np.round(rng.uniform(0.1, 0.9), 2))
        when = _random_condition(neighborhood_type, num_states, rng)

        compiled_rules.append({
            "from_codes": from_codes,
            "to_code": to_code,
            "set": {},
            "when": when,
            "probability": prob,
        })

    return {
        "format_version": "2.0",
        "kind": "cells",
        "name": f"Random_{neighborhood_type}_{num_states}states",
        "state_codes": {f"S{i}"  for i in range(num_states)},
        "state_names": [f"S{i}" for i in range(num_states)],
        "neighbor_offsets": offsets,
        "world": {
            "boundary": boundary,
            "outside_code": outside_code,
            "outside_values": {},
        },
        "compiled_rules": compiled_rules,
    }