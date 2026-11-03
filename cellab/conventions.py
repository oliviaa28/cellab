"""Conventions shared by the loader and the validator (contract, section 5.2).

Everything here is written once, so the loader and the validator can never
disagree about, for example, the order of the neighbors.
"""

from __future__ import annotations

from pathlib import Path

# The project folder, found from the location of this file
# (cellab/conventions.py -> its parent is cellab/, the parent of that is the project).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

FORMAT_VERSION = "2.0"

# Neighbor offsets as (row change, column change), in the contract order.
# Row 0 is at the top, so N (up) is row - 1.
NEIGHBOR_OFFSETS = {
    # NW, N, NE, W, E, SW, S, SE
    "moore": [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
    # N, W, E, S
    "von_neumann": [(-1, 0), (0, -1), (0, 1), (1, 0)],
    # W, E
    "1d": [(0, -1), (0, 1)],
}


def json_pointer(path) -> str:
    """Turn a path like ["rules", 2, "when"] into the JSON Pointer "/rules/2/when".

    The empty path (the whole file) is the empty string, as in the standard.
    Inside a name, "~" is written "~0" and "/" is written "~1".
    """
    text = ""
    for part in path:
        part = str(part).replace("~", "~0").replace("/", "~1")
        text = text + "/" + part
    return text


def merge_parameters(raw_automaton: dict, raw_experiment: dict | None) -> dict:
    """Final parameter values: the automaton's, overridden by the experiment's."""
    merged = dict(raw_automaton.get("parameters", {}))
    if raw_experiment is not None:
        for name, value in raw_experiment.get("parameters", {}).items():
            merged[name] = value
    return merged