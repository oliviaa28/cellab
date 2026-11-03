"""Shapes of the dictionaries passed between CelLab modules (contract, sections 5.3-5.5).

Named datatypes.py (not types.py) so it never clashes with Python's
built-in "types" module.

These classes do nothing at runtime. They only describe which keys a
dictionary has and what type each value is, so the editor can warn us
when a key is misspelled.

Rule from the contract: optional fields that are missing in the JSON file
appear empty ([] or {}) in the loaded dictionaries, never missing.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np


# ---------- errors (5.5) ----------

class ValidationIssue(TypedDict):
    level: str      # "error" or "warning"
    code: str       # for example "UNKNOWN_STATE"
    where: str      # JSON Pointer, for example "/rules/2/when/count"
    message: str    # in English


# ---------- loaded automaton (5.4) ----------

class CompiledRule(TypedDict):
    from_codes: list[int]           # codes of the starting states (all codes for "*")
    to_code: int | None             # new state, or None if the state does not change
    set: dict[str, str]             # variable name -> formula text
    when: dict[str, Any] | None     # state names -> codes, parameters -> numbers
    probability: float              # parameters already replaced by numbers


class World(TypedDict):
    boundary: str                   # "fixed" or "toroidal"
    outside_code: int | None        # code of the outside state; None for "toroidal"
    outside_values: dict[str, float]  # variable name -> value outside the grid


class LoadedAutomaton(TypedDict):
    # fields copied from the JSON file (missing optional ones become empty)
    format_version: str
    kind: str                       # "cells" or "agents"
    name: str
    description: str
    states: list[dict[str, Any]]    # original list: name, symbol, color
    variables: list[dict[str, Any]]  # original list: name, type, default
    neighborhood: dict[str, Any]    # original: type, radius
    rules: list[dict[str, Any]]     # original rules, kept for error messages

    # keys added by the loader
    state_codes: dict[str, int]     # {"OFF": 0, "FIRING": 1, "DYING": 2}
    state_names: list[str]          # ["OFF", "FIRING", "DYING"]
    symbols: dict[str, int]         # {".": 0, "F": 1, "d": 2}
    colors: list[str | None]        # one per state, None where missing
    variable_types: dict[str, str]  # name -> "int" or "float"
    variable_defaults: dict[str, float]  # name -> default value
    parameters: dict[str, float]    # final values: automaton, overridden by experiment
    neighbor_offsets: list[tuple[int, int]]  # in the contract order (5.2)
    world: World
    compiled_rules: list[CompiledRule]

    # agent fields, copied as they are (Phase 3: details finalized by Person 4)
    agent_types: list[dict[str, Any]]   # [] for kind "cells"
    agent_order: str                    # "fixed" if missing
    agent_rules: list[dict[str, Any]]   # [] for kind "cells"


# ---------- loaded experiment (6.6) ----------
# PROPOSAL: the contract does not fix this shape yet; to be confirmed with P3.

class LoadedExperiment(TypedDict):
    format_version: str
    automaton: str                  # path to the automaton file
    parameters: dict[str, float]    # {} if missing
    world: dict[str, Any]           # rows, cols, boundary, outside
    initial_state: dict[str, Any]   # cells, values, agents
    generations: int
    storage: dict[str, Any]         # {"mode": "all"} / {"mode": "last"} / {"mode": "every", "every": k}
    seed: int | None                # None if missing


# ---------- state (5.3) ----------

class Agent(TypedDict):
    id: int                         # unique in the whole experiment
    type: str
    row: int
    col: int
    attributes: dict[str, Any]


class State(TypedDict):
    cells: np.ndarray               # integers, rows x cols: the state code of each cell
    values: dict[str, np.ndarray]   # variable name -> matrix of values; {} if none
    agents: list[Agent]             # [] if none
    generation: int


# ---------- result of loader.load_files ----------

class LoadResult(TypedDict):
    automaton: LoadedAutomaton | None     # None if there is any error
    experiment: LoadedExperiment | None   # None if there is any error
    automaton_issues: list[ValidationIssue]
    experiment_issues: list[ValidationIssue]