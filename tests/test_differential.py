"""Differential tests: vectorized engine vs reference engine."""

import numpy as np
import pytest

from cellab.engine_cells import step_cells
from cellab.initial_state import create_initial_state
from cellab.loader import load_files
from cellab.random_automata import generate_random_automaton, generate_random_case
from cellab.reference_engine import step_cells_reference
from cellab.validator import has_errors, validate_automaton


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
    case = generate_random_case(rng_gen)
    aut = case["automaton"]
    state = create_initial_state(aut, case["experiment"], rng_gen)

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


@pytest.mark.parametrize("experiment_path", [
    "experiments/forest_fire_manual.json",
    "experiments/forest_fire_stochastic_random.json",
    "experiments/brians_brain_random.json",
    "experiments/rule_30_center.json",
    "experiments/cyclic_random.json",
])
def test_differential_real_automata(experiment_path: str):
    """The two engines agree on the real automata files for 30 generations."""
    loaded = load_files(experiment_path)
    aut = loaded["automaton"]
    state = create_initial_state(aut, loaded["experiment"], np.random.default_rng(1))

    s_fast = state
    s_ref = state
    rng_fast = np.random.default_rng(5)
    rng_ref = np.random.default_rng(5)
    for gen in range(30):
        s_fast = step_cells(aut, s_fast, rng_fast)
        s_ref = step_cells_reference(aut, s_ref, rng_ref)
        assert np.array_equal(s_fast["cells"], s_ref["cells"]), f"Mismatch at gen {gen+1}"


@pytest.mark.parametrize("neighborhood_type", ["moore", "von_neumann", "1d"])
def test_random_automata_are_valid(neighborhood_type: str):
    """Generated automata pass the validator, for every neighborhood."""
    rng = np.random.default_rng(123)
    for _ in range(50):
        raw_aut = generate_random_automaton(rng, neighborhood_type)
        assert not has_errors(validate_automaton(raw_aut))


def test_random_case_is_loaded_like_a_real_file():
    """A generated case has the same loaded shape as a real automaton."""
    case = generate_random_case(np.random.default_rng(7), "moore")
    aut = case["automaton"]
    real = load_files("experiments/brians_brain_random.json")["automaton"]

    assert set(aut.keys()) == set(real.keys())
    for code, name in enumerate(aut["state_names"]):
        assert aut["state_codes"][name] == code


def test_reference_engine_rejects_set_formulas():
    """Rules with "set" are Phase 2: the reference engine must not ignore them."""
    automaton = {
        "neighbor_offsets": [(0, -1), (0, 1)],
        "world": {"boundary": "toroidal", "outside_code": None, "outside_values": {}},
        "compiled_rules": [
            {"from_codes": [0], "to_code": None, "set": {"x": "x + 1"},
             "when": None, "probability": 1.0},
        ],
    }
    state = {"cells": np.zeros((1, 3), dtype=np.int64), "generation": 0}
    with pytest.raises(NotImplementedError):
        step_cells_reference(automaton, state, np.random.default_rng(0))