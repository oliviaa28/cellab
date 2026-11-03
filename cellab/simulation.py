"""simulation.py – Run the cellular automaton simulation.

Phase 1: creates the RNG from the seed, selects the right engine by
``kind``, runs N generations, and returns results.

Key design rules (from the contract):
- A single ``np.random.default_rng(seed)`` is created here and passed
  everywhere.  Nobody else creates random generators.
- ``step`` picks the engine by ``automaton["kind"]``.
- ``run`` calls ``step`` N times and collects states according to storage
  mode.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cellab.engine_cells import step_cells


# ---------------------------------------------------------------------------
# Step (one generation)
# ---------------------------------------------------------------------------

def step(
    automaton: dict[str, Any],
    state: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Advance the automaton by one generation and return a *new* state.

    Dispatches to the correct engine based on ``automaton["kind"]``.
    """
    kind = automaton.get("kind", "cells")
    if kind == "cells":
        return step_cells(automaton, state, rng)
    elif kind == "agents":
        # Phase 3 – agent engine
        # For now, we only run the cell rules.
        # When engine_agents.py is ready, import step_agents and call it.
        return step_cells(automaton, state, rng)
    else:
        raise ValueError(f"Unknown automaton kind: {kind!r}")


# ---------------------------------------------------------------------------
# Run (multiple generations)
# ---------------------------------------------------------------------------

def run(
    automaton: dict[str, Any],
    initial_state: dict[str, Any],
    generations: int,
    storage: dict[str, Any],
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Simulate *generations* steps and return the results.

    Parameters
    ----------
    automaton : loaded automaton dict
    initial_state : generation-0 state
    generations : how many steps to simulate
    storage : ``{"mode": "all"}`` | ``{"mode": "last"}`` | ``{"mode": "every", "every": k}``
    seed : random seed (creates a new rng); ignored if *rng* is given
    rng : existing generator (for resuming a saved run)

    Returns
    -------
    dict with keys:
        ``states`` – list of stored states (always includes the last one)
        ``rng`` – the generator after the run (for saving / resuming)
        ``last_state`` – the final state
    """
    if rng is None:
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

    mode = storage["mode"]
    every = storage.get("every", 1)

    stored_states: list[dict[str, Any]] = []

    # Optionally store generation 0
    if mode == "all" or (mode == "every" and 0 % every == 0):
        stored_states.append(initial_state)

    current = initial_state
    for gen_i in range(1, generations + 1):
        current = step(automaton, current, rng)

        if mode == "all":
            stored_states.append(current)
        elif mode == "every" and gen_i % every == 0:
            stored_states.append(current)
        # "last" mode: we store nothing during the loop

    # Always include the final state in stored_states if not already there
    if mode == "last":
        stored_states.append(current)
    elif mode == "every" and generations % every != 0:
        stored_states.append(current)

    return {
        "states": stored_states,
        "rng": rng,
        "last_state": current,
    }
