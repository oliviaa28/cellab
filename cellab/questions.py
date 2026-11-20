"""questions.py – The 5 questions about a simulation.

    1. What is the state after N generations?            state_after
    2. When does the system become stable?               when_stable
    3. Does a given configuration ever appear?           find_configuration
    4. How many cells of type X are there after N?       count_after
    5. Does the simulation enter a cycle?                find_cycle

The shortcut: every state seen is remembered by its fingerprint (storage.fingerprint).
When a fingerprint comes back, the simulation is in a cycle: state i == state j.
For a deterministic automaton everything after that repeats with period j - i,
so the state at generation N is found without simulating all N generations.

Deterministic or not: if the automaton uses randomness while it runs (probabilities,
random(), agents in random order...), the same grid does not mean the same future,
because the random generator has moved on. For such automata the answers about
cycles and stability are only what was observed ("certain": False), and
state_after always simulates every generation.

Every function takes the loaded automaton, a starting state and the rng that the
simulation uses (the same one simulation.run would get), and never changes the state.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from cellab.expressions import rule_formulas, uses_random
from cellab.simulation import step
from cellab.storage import fingerprint

DEFAULT_LIMIT = 10_000


# ---------------------------------------------------------------------------
# Deterministic or not
# ---------------------------------------------------------------------------

def is_deterministic(automaton: dict[str, Any]) -> bool:
    """True if the next state depends only on the current one (no randomness while running)."""
    for rule in automaton["compiled_rules"]:
        if rule["probability"] < 1:
            return False
        for text in rule_formulas(rule):
            if uses_random(text):
                return False
    if automaton.get("agent_order") == "random" and automaton.get("kind") == "agents":
        return False
    # agent actions such as {"move": "random_free"} or random() in their formulas
    if "random" in json.dumps(automaton.get("agent_rules", [])):
        return False
    return True


# ---------------------------------------------------------------------------
# Walking through the generations, remembering every fingerprint
# ---------------------------------------------------------------------------

def _walk(automaton, state, rng, limit, stop=None) -> dict[str, Any]:
    """Simulate up to `limit` generations from `state`.

    Stops early when stop(state) is true, or when a state comes back (deterministic automata).
    Returns {"state": the last state, "steps": generations simulated,
             "repeat": (first generation, generation it came back) or None, "stopped": bool}.
    """
    deterministic = is_deterministic(automaton)
    seen = {fingerprint(state): 0}
    current = state
    for steps in range(1, limit + 1):
        current = step(automaton, current, rng)
        if stop is not None and stop(current):
            return {"state": current, "steps": steps, "repeat": None, "stopped": True}
        key = fingerprint(current)
        if key in seen and deterministic:
            return {"state": current, "steps": steps, "repeat": (seen[key], steps), "stopped": False}
        seen.setdefault(key, steps)
    return {"state": current, "steps": limit, "repeat": None, "stopped": False}


def _relative(state: dict[str, Any], generation: int) -> int:
    """A step count from `state` turned into a generation number."""
    return state.get("generation", 0) + generation


# ---------------------------------------------------------------------------
# Question 1: the state after N generations
# ---------------------------------------------------------------------------

def state_after(automaton, state, generations: int, rng) -> dict[str, Any]:
    """The state `generations` steps after `state`.

    Returns {"state", "generation", "simulated": steps really computed, "shortcut": bool}.
    """
    if generations < 0:
        raise ValueError("generations must be 0 or more")
    if generations == 0:
        return {"state": state, "generation": _relative(state, 0), "simulated": 0, "shortcut": False}

    walk = _walk(automaton, state, rng, generations)
    if walk["repeat"] is None:
        return {"state": walk["state"], "generation": _relative(state, generations),
                "simulated": walk["steps"], "shortcut": False}

    # state at `first` == state at `again`: from here on everything repeats with this period
    first, again = walk["repeat"]
    period = again - first
    remaining = (generations - again) % period
    current = walk["state"]
    for _ in range(remaining):
        current = step(automaton, current, rng)
    current = dict(current, generation=_relative(state, generations))
    return {"state": current, "generation": _relative(state, generations),
            "simulated": walk["steps"] + remaining, "shortcut": True}


# ---------------------------------------------------------------------------
# Question 4: how many cells of type X after N generations
# ---------------------------------------------------------------------------

def count_after(automaton, state, generations: int, state_name: str, rng) -> dict[str, Any]:
    """How many cells are in state `state_name` after `generations` steps."""
    if state_name not in automaton["state_codes"]:
        raise ValueError(f"State '{state_name}' is not defined "
                         f"(defined: {', '.join(automaton['state_names'])}).")
    answer = state_after(automaton, state, generations, rng)
    code = automaton["state_codes"][state_name]
    count = int(np.sum(answer["state"]["cells"] == code))
    return {"count": count, "state_name": state_name, "generation": answer["generation"],
            "total": int(answer["state"]["cells"].size), "shortcut": answer["shortcut"]}


def count_all(automaton, state: dict[str, Any]) -> dict[str, int]:
    """How many cells are in each state, right now: {"EMPTY": 10, "TREE": 5, ...}."""
    counts = {}
    for name, code in automaton["state_codes"].items():
        counts[name] = int(np.sum(state["cells"] == code))
    return counts


# ---------------------------------------------------------------------------
# Question 5: does the simulation enter a cycle
# ---------------------------------------------------------------------------

def find_cycle(automaton, state, rng, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    """The first cycle: {"cycle": bool, "start": generation, "length": period, "certain": bool}.

    A stable state is a cycle of length 1.
    Deterministic automaton: a repeated state is a sure cycle.
    Random automaton: only an observed repetition of the grid ("certain": False).
    """
    if not is_deterministic(automaton):
        return _observed_repetition(automaton, state, rng, limit)

    walk = _walk(automaton, state, rng, limit)
    if walk["repeat"] is None:
        return {"cycle": False, "start": None, "length": None, "certain": False, "checked": limit}
    first, again = walk["repeat"]
    return {"cycle": True, "start": _relative(state, first), "length": again - first,
            "certain": True, "checked": again}


def _observed_repetition(automaton, state, rng, limit) -> dict[str, Any]:
    """For random automata: the first grid that is seen again (not a guaranteed cycle)."""
    seen = {fingerprint(state): 0}
    current = state
    for steps in range(1, limit + 1):
        current = step(automaton, current, rng)
        key = fingerprint(current)
        if key in seen:
            return {"cycle": True, "start": _relative(state, seen[key]), "length": steps - seen[key],
                    "certain": False, "checked": steps}
        seen[key] = steps
    return {"cycle": False, "start": None, "length": None, "certain": False, "checked": limit}


# ---------------------------------------------------------------------------
# Question 2: when does the system become stable
# ---------------------------------------------------------------------------

def when_stable(automaton, state, rng, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    """The first generation after which nothing changes any more.

    Returns {"stable": bool, "generation": g or None, "certain": bool, "reason": text}.
    """
    if is_deterministic(automaton):
        cycle = find_cycle(automaton, state, rng, limit)
        if not cycle["cycle"]:
            return {"stable": False, "generation": None, "certain": False,
                    "reason": f"no repeated state in {limit} generations"}
        if cycle["length"] == 1:
            return {"stable": True, "generation": cycle["start"], "certain": True,
                    "reason": "the state does not change any more"}
        return {"stable": False, "generation": None, "certain": True,
                "reason": f"it enters a cycle of length {cycle['length']} at generation {cycle['start']}"}

    # random automaton: two equal generations in a row, as observed
    previous = fingerprint(state)
    current = state
    for steps in range(1, limit + 1):
        current = step(automaton, current, rng)
        key = fingerprint(current)
        if key == previous:
            return {"stable": True, "generation": _relative(state, steps - 1), "certain": False,
                    "reason": "two equal generations in a row (randomness could still change it)"}
        previous = key
    return {"stable": False, "generation": None, "certain": False,
            "reason": f"it kept changing for {limit} generations"}


# ---------------------------------------------------------------------------
# Question 3: does a configuration ever appear
# ---------------------------------------------------------------------------

def pattern_from_symbols(automaton, rows: list[str]) -> np.ndarray:
    """A configuration written with the state symbols, e.g. [".F.", "FFF"] -> matrix of codes."""
    codes = []
    for row in rows:
        line = []
        for symbol in row:
            if symbol not in automaton["symbols"]:
                raise ValueError(f"Symbol '{symbol}' is not defined "
                                 f"(defined: {', '.join(automaton['symbols'])}).")
            line.append(automaton["symbols"][symbol])
        codes.append(line)
    if len({len(line) for line in codes}) != 1:
        raise ValueError("Every row of the configuration must have the same length.")
    return np.array(codes, dtype=np.int64)


def where_is(pattern: np.ndarray, cells: np.ndarray) -> tuple[int, int] | None:
    """The top-left position of the first place where `pattern` appears in `cells`, or None."""
    pattern = np.asarray(pattern)
    if pattern.shape[0] > cells.shape[0] or pattern.shape[1] > cells.shape[1]:
        return None
    windows = sliding_window_view(cells, pattern.shape)
    matches = np.argwhere((windows == pattern).all(axis=(2, 3)))
    if len(matches) == 0:
        return None
    return int(matches[0][0]), int(matches[0][1])


def find_configuration(automaton, state, pattern, rng, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    """The first generation where `pattern` (a matrix of state codes) appears anywhere in the grid.

    A pattern as large as the grid means the whole grid must be equal.
    Returns {"found": bool, "generation", "position", "certain": bool}.
    "certain" for a "not found" answer: the automaton is deterministic and entered a cycle,
    so every future state was already checked.
    """
    pattern = np.asarray(pattern)
    position = where_is(pattern, state["cells"])
    if position is not None:
        return {"found": True, "generation": _relative(state, 0), "position": position, "certain": True}

    walk = _walk(automaton, state, rng, limit, stop=lambda current: where_is(pattern, current["cells"]) is not None)
    if walk["stopped"]:
        return {"found": True, "generation": _relative(state, walk["steps"]),
                "position": where_is(pattern, walk["state"]["cells"]), "certain": True}
    return {"found": False, "generation": None, "position": None,
            "certain": walk["repeat"] is not None}
