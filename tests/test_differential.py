"""Differential tests: vectorized engine vs reference engine."""

import numpy as np
import pytest

from cellab.engine_cells import step_cells
from cellab.random_automata import generate_random_automaton
from cellab.reference_engine import step_cells_reference


def test_rule_30_hand_calculated():
    """Manual test for 1D Rule 30."""
    offsets = [(0, -1), (0, 1)]  # [W, E]
    rules = [
        {"from_codes": [0, 1], "to_code": 0, "when": {"pattern": [1, 1, 1]}, "probability": 1.0},
        {"from_codes": [0, 1], "to_code": 0, "when": {"pattern": [1, 1, 0]}, "probability": 1.0},
        {"from_codes": [0, 1], "to_code": 0, "when": {"pattern": [1, 0, 1]}, "probability": 1.0},
        {"from_codes": [0, 1], "to_code": 1, "when": {"pattern": [1, 0, 0]}, "probability": 1.0},
        {"from_codes": [0, 1], "to_code": 1, "when": {"pattern": [0, 1, 1]}, "probability": 1.0},
        {"from_codes": [0, 1], "to_code": 1, "when": {"pattern": [0, 1, 0]}, "probability": 1.0},
        {"from_codes": [0, 1], "to_code": 1, "when": {"pattern": [0, 0, 1]}, "probability": 1.0},
        {"from_codes": [0, 1], "to_code": 0, "when": {"pattern": [0, 0, 0]}, "probability": 1.0},
    ]
    automaton = {
        "neighbor_offsets": offsets,
        "world": {"boundary": "fixed", "outside_code": 0, "outside_values": {}},
        "compiled_rules": rules,
    }

    # Initial row with a single 1 in the middle
    row = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0]])
    state = {"cells": row, "generation": 0}

    # Step 1 hand-calculated: ... 0, 1, 1, 1, 0 ...
    expected_gen1 = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0]])
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    res_vectorized = step_cells(automaton, state, rng1)
    res_reference = step_cells_reference(automaton, state, rng2)

    assert np.array_equal(res_vectorized["cells"], expected_gen1)
    assert np.array_equal(res_reference["cells"], expected_gen1)


@pytest.mark.parametrize("seed", range(1, 101))
def test_differential_random_automata(seed: int):
    """Run differential testing on 100 randomly generated cellular automata."""
    rng_gen = np.random.default_rng(seed)
    aut = generate_random_automaton(rng_gen)

    # Grids: 1D for 1d neighborhood, 2D otherwise
    is_1d = len(aut["neighbor_offsets"]) == 2
    num_states = len(aut["state_codes"])
    rows = 1 if is_1d else int(rng_gen.integers(4, 15))
    cols = int(rng_gen.integers(5, 20))

    init_grid = rng_gen.integers(0, num_states, size=(rows, cols))
    state = {"cells": init_grid, "generation": 0}

    # Test over 5 generations
    s_fast = state
    s_ref = state
    rng_fast = np.random.default_rng(seed * 1000 + 7)
    rng_ref = np.random.default_rng(seed * 1000 + 7)

    for gen in range(5):
        s_fast = step_cells(aut, s_fast, rng_fast)
        s_ref = step_cells_reference(aut, s_ref, rng_ref)

        assert np.array_equal(
            s_fast["cells"], s_ref["cells"]
        ), f"Mismatch at gen {gen+1} with seed {seed} on automaton {aut['name']}"