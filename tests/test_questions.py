"""Tests for the 5 questions (cellab.questions) and the fingerprints (cellab.storage)."""

import numpy as np
import pytest

from cellab.initial_state import create_initial_state
from cellab.loader import load_files
from cellab.questions import (
    count_after,
    count_all,
    find_configuration,
    find_cycle,
    is_deterministic,
    pattern_from_symbols,
    state_after,
    when_stable,
    where_is,
)
from cellab.simulation import run
from cellab.storage import fingerprint, load_run, save_run


def load(experiment_path):
    loaded = load_files(experiment_path)
    assert loaded["automaton"] is not None, loaded
    experiment = loaded["experiment"]
    state = create_initial_state(loaded["automaton"], experiment, np.random.default_rng(experiment["seed"] or 0))
    return loaded["automaton"], state


def run_direct(automaton, state, generations, seed=0):
    return run(automaton, state, generations, {"mode": "last"}, rng=np.random.default_rng(seed))["last_state"]


# ---------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------

def test_same_state_same_fingerprint_whatever_the_generation():
    state = {"cells": np.array([[0, 1], [1, 0]]), "values": {"x": np.array([[0.0, -0.0], [1.5, 2.0]])},
             "agents": [], "generation": 3}
    same = {"cells": state["cells"].copy(), "values": {"x": np.array([[0.0, 0.0], [1.5, 2.0]])},
            "agents": [], "generation": 99}
    assert fingerprint(state) == fingerprint(same)
    assert len(fingerprint(state)) == 32


def test_any_change_changes_the_fingerprint():
    base = {"cells": np.array([[0, 1], [1, 0]]), "values": {"x": np.zeros((2, 2))}, "agents": [], "generation": 0}
    other_cell = dict(base, cells=np.array([[1, 1], [1, 0]]))
    other_value = dict(base, values={"x": np.array([[0.0, 0.0], [0.0, 0.5]])})
    other_agent = dict(base, agents=[{"id": 1, "type": "ANT", "row": 0, "col": 0, "attributes": {}}])
    other_shape = dict(base, cells=np.array([[0, 1, 1, 0]]))
    fingerprints = {fingerprint(s) for s in [base, other_cell, other_value, other_agent, other_shape]}
    assert len(fingerprints) == 5


def test_saved_run_has_fingerprints(tmp_path, monkeypatch):
    automaton, state = load("experiments/forest_fire_manual.json")
    result = run(automaton, state, 3, {"mode": "all"}, rng=np.random.default_rng(0))
    monkeypatch.setattr("cellab.storage._project_root", lambda: tmp_path)
    run_dir = save_run("test", {"name": "x"}, {"generations": 3}, result)
    meta = load_run(run_dir)["meta"]
    assert meta["fingerprints"]["0"] == fingerprint(state)
    assert meta["fingerprints"]["3"] == fingerprint(result["last_state"])


# ---------------------------------------------------------------------
# Deterministic or not
# ---------------------------------------------------------------------

@pytest.mark.parametrize("experiment_path, expected", [
    ("experiments/brians_brain_random.json", True),       # random start, but deterministic rules
    ("experiments/sandpile_manual.json", True),
    ("experiments/forest_fire_stochastic_random.json", False),
    ("experiments/epidemic_stochastic_random.json", False),   # random() in a formula
])
def test_is_deterministic(experiment_path, expected):
    automaton, _ = load(experiment_path)
    assert is_deterministic(automaton) is expected


# ---------------------------------------------------------------------
# Questions 1 and 4: state and counts after N generations
# ---------------------------------------------------------------------

def test_state_after_uses_the_cycle_and_matches_the_full_simulation():
    # Rule 184 on a ring: after 2 generations the cars move together, with period 10.
    automaton, state = load("experiments/traffic_184_road.json")
    answer = state_after(automaton, state, 1_000, np.random.default_rng(0))
    assert answer["shortcut"]
    assert answer["simulated"] < 30
    assert answer["generation"] == 1_000
    np.testing.assert_array_equal(answer["state"]["cells"], run_direct(automaton, state, 1_000)["cells"])


def test_state_after_a_random_automaton_simulates_everything():
    automaton, state = load("experiments/forest_fire_stochastic_random.json")
    answer = state_after(automaton, state, 20, np.random.default_rng(4))
    assert not answer["shortcut"] and answer["simulated"] == 20
    np.testing.assert_array_equal(answer["state"]["cells"], run_direct(automaton, state, 20, seed=4)["cells"])


def test_count_after_forest_fire_by_hand():
    # 5x5 forest with fire in the middle: after 1 generation 8 cells burn, after 3 all is empty.
    automaton, state = load("experiments/forest_fire_manual.json")
    rng = np.random.default_rng(0)
    assert count_after(automaton, state, 1, "FIRE", rng)["count"] == 8
    assert count_after(automaton, state, 3, "EMPTY", rng)["count"] == 25
    assert count_all(automaton, state) == {"EMPTY": 0, "TREE": 24, "FIRE": 1}
    with pytest.raises(ValueError):
        count_after(automaton, state, 1, "WATER", rng)


# ---------------------------------------------------------------------
# Questions 2 and 5: stable state and cycles
# ---------------------------------------------------------------------

def test_forest_fire_becomes_stable_at_generation_3():
    automaton, state = load("experiments/forest_fire_manual.json")
    answer = when_stable(automaton, state, np.random.default_rng(0))
    assert answer == {"stable": True, "generation": 3, "certain": True,
                      "reason": "the state does not change any more"}


def test_traffic_on_a_ring_is_a_cycle_not_a_stable_state():
    automaton, state = load("experiments/traffic_184_road.json")
    cycle = find_cycle(automaton, state, np.random.default_rng(0))
    assert cycle["cycle"] and cycle["certain"]
    assert (cycle["start"], cycle["length"]) == (2, 10)
    stable = when_stable(automaton, state, np.random.default_rng(0))
    assert not stable["stable"] and stable["certain"]


def test_cycle_found_is_a_real_cycle():
    automaton, state = load("experiments/cyclic_random.json")
    cycle = find_cycle(automaton, state, np.random.default_rng(0))
    assert cycle["cycle"]
    at_start = run_direct(automaton, state, cycle["start"])
    one_period_later = run_direct(automaton, state, cycle["start"] + cycle["length"])
    np.testing.assert_array_equal(at_start["cells"], one_period_later["cells"])


def test_no_cycle_within_the_limit():
    automaton, state = load("experiments/brians_brain_random.json")
    answer = find_cycle(automaton, state, np.random.default_rng(0), limit=20)
    assert answer == {"cycle": False, "start": None, "length": None, "certain": False, "checked": 20}


def test_random_automaton_answers_are_not_certain():
    automaton, state = load("experiments/epidemic_stochastic_random.json")
    answer = when_stable(automaton, state, np.random.default_rng(7), limit=500)
    assert answer["certain"] is False


# ---------------------------------------------------------------------
# Question 3: does a configuration appear
# ---------------------------------------------------------------------

def test_where_is_finds_a_small_pattern():
    cells = np.array([[0, 0, 0, 0],
                      [0, 1, 2, 0],
                      [0, 2, 1, 0]])
    assert where_is(np.array([[1, 2], [2, 1]]), cells) == (1, 1)
    assert where_is(np.array([[2, 2]]), cells) is None
    assert where_is(np.zeros((5, 5)), cells) is None


def test_forest_fire_ring_of_fire_appears_at_generation_1():
    automaton, state = load("experiments/forest_fire_manual.json")
    ring = pattern_from_symbols(automaton, ["FFF", "F.F", "FFF"])
    answer = find_configuration(automaton, state, ring, np.random.default_rng(0))
    assert answer == {"found": True, "generation": 1, "position": (1, 1), "certain": True}


def test_a_configuration_already_there_is_found_at_the_start():
    automaton, state = load("experiments/forest_fire_manual.json")
    trees = pattern_from_symbols(automaton, ["TTT"])
    assert find_configuration(automaton, state, trees, np.random.default_rng(0))["generation"] == 0


def test_a_configuration_that_never_appears_is_certain_after_the_cycle():
    # the forest becomes empty and stays so: a grid full of fire can never come
    automaton, state = load("experiments/forest_fire_manual.json")
    all_fire = pattern_from_symbols(automaton, ["FFFFF"] * 5)
    answer = find_configuration(automaton, state, all_fire, np.random.default_rng(0))
    assert answer == {"found": False, "generation": None, "position": None, "certain": True}


def test_pattern_from_symbols_checks_the_symbols():
    automaton, _ = load("experiments/forest_fire_manual.json")
    with pytest.raises(ValueError):
        pattern_from_symbols(automaton, ["TX"])
    with pytest.raises(ValueError):
        pattern_from_symbols(automaton, ["TT", "T"])
