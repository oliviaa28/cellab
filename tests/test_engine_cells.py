"""Tests for engine_cells.step_cells.

The automata below are written by hand in the *loaded* form from
docs/contract.md (compiled_rules, neighbor_offsets, world). When the loader
is ready, these helpers should be replaced by loading the real JSON files.
"""

import numpy as np
import pytest

from cellab.engine_cells import step_cells

MOORE = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
LINE = [(0, -1), (0, 1)]


def make_state(cells):
    return {"cells": np.array(cells, dtype=np.int64), "values": {}, "agents": [], "generation": 0}


def make_rule(from_codes, to_code, when=None, probability=1.0):
    return {"from_codes": from_codes, "to_code": to_code, "set": {}, "when": when, "probability": probability}


def make_automaton(rules, offsets, boundary="fixed", outside_code=0):
    return {
        "compiled_rules": rules,
        "neighbor_offsets": offsets,
        "world": {"boundary": boundary, "outside_code": outside_code, "outside_values": {}},
    }


def rows_to_cells(rows):
    return [[int(ch) for ch in row] for row in rows]


# ---------- Forest Fire: EMPTY=0, TREE=1, FIRE=2 ----------

FOREST_FIRE = make_automaton(
    [
        make_rule([1], 2, {"count": 2, "min": 1}),
        make_rule([2], 0),
    ],
    MOORE,
    boundary="fixed",
    outside_code=0,
)


def test_forest_fire_5x5_matches_hand_calculation():
    state = make_state(rows_to_cells(["11111", "11111", "11211", "11111", "11111"]))
    rng = np.random.default_rng(0)

    state = step_cells(FOREST_FIRE, state, rng)
    np.testing.assert_array_equal(state["cells"], rows_to_cells(["11111", "12221", "12021", "12221", "11111"]))

    state = step_cells(FOREST_FIRE, state, rng)
    np.testing.assert_array_equal(state["cells"], rows_to_cells(["22222", "20002", "20002", "20002", "22222"]))

    state = step_cells(FOREST_FIRE, state, rng)
    np.testing.assert_array_equal(state["cells"], np.zeros((5, 5)))

    stable = step_cells(FOREST_FIRE, state, rng)
    np.testing.assert_array_equal(stable["cells"], state["cells"])
    assert stable["generation"] == 4


# ---------- Brian's Brain: OFF=0, FIRING=1, DYING=2 ----------

BRIANS_BRAIN = make_automaton(
    [
        make_rule([1], 2),
        make_rule([2], 0),
        make_rule([0], 1, {"count": 1, "min": 2, "max": 2}),
    ],
    MOORE,
    boundary="fixed",
    outside_code=0,
)


def test_brians_brain_fires_with_exactly_two_firing_neighbors():
    two = make_state([[1, 0, 1], [0, 0, 0], [0, 0, 0]])
    three = make_state([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
    rng = np.random.default_rng(0)

    assert step_cells(BRIANS_BRAIN, two, rng)["cells"][1, 1] == 1
    assert step_cells(BRIANS_BRAIN, three, rng)["cells"][1, 1] == 0


def test_brians_brain_firing_becomes_dying_then_off():
    state = make_state([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    rng = np.random.default_rng(0)

    state = step_cells(BRIANS_BRAIN, state, rng)
    assert state["cells"][1, 1] == 2
    state = step_cells(BRIANS_BRAIN, state, rng)
    assert state["cells"][1, 1] == 0


# ---------- Rule 30: OFF=0, ON=1 ----------

def rule_30():
    rules = []
    for index in range(8):
        left = (index >> 2) & 1
        center = (index >> 1) & 1
        right = index & 1
        rules.append(make_rule([0, 1], (30 >> index) & 1, {"pattern": [left, center, right]}))
    return make_automaton(rules, LINE, boundary="fixed", outside_code=0)


def test_rule_30_first_generations_match_known_triangle():
    expected = ["00000100000", "00001110000", "00011001000", "00110111100", "01100100010"]
    automaton = rule_30()
    state = make_state([[int(ch) for ch in expected[0]]])
    rng = np.random.default_rng(0)

    for row in expected[1:]:
        state = step_cells(automaton, state, rng)
        np.testing.assert_array_equal(state["cells"], [[int(ch) for ch in row]])


# ---------- boundaries ----------

def test_fixed_boundary_uses_outside_code():
    automaton = make_automaton([make_rule([0], 1, {"count": 1, "min": 2})], LINE, boundary="fixed", outside_code=1)
    result = step_cells(automaton, make_state([[0]]), np.random.default_rng(0))
    np.testing.assert_array_equal(result["cells"], [[1]])


def test_toroidal_boundary_wraps_around():
    # The cell at the right edge sees the left edge as its east neighbor.
    automaton = make_automaton([make_rule([0], 1, {"pattern": [0, 0, 1]})], LINE, boundary="toroidal")
    result = step_cells(automaton, make_state([[1, 0, 0]]), np.random.default_rng(0))
    np.testing.assert_array_equal(result["cells"], [[1, 0, 1]])


def test_missing_world_is_an_error():
    automaton = {"compiled_rules": [], "neighbor_offsets": LINE}
    with pytest.raises(KeyError):
        step_cells(automaton, make_state([[0]]), np.random.default_rng(0))


# ---------- rule semantics ----------

def test_first_matching_rule_wins_and_unmatched_cells_stay():
    automaton = make_automaton([make_rule([0], 1), make_rule([0], 2)], LINE)
    result = step_cells(automaton, make_state([[0, 3]]), np.random.default_rng(0))
    np.testing.assert_array_equal(result["cells"], [[1, 3]])


def test_update_is_synchronous_and_input_is_not_modified():
    # Every ON cell turns OFF and every OFF cell next to an ON cell turns ON.
    automaton = make_automaton(
        [make_rule([1], 0), make_rule([0], 1, {"count": 1, "min": 1})], LINE
    )
    initial = make_state([[0, 1, 0, 0]])
    result = step_cells(automaton, initial, np.random.default_rng(0))

    np.testing.assert_array_equal(result["cells"], [[1, 0, 1, 0]])
    np.testing.assert_array_equal(initial["cells"], [[0, 1, 0, 0]])


# ---------- randomness ----------

def test_same_seed_gives_same_result():
    automaton = make_automaton([make_rule([0], 1, probability=0.5)], MOORE, boundary="toroidal")
    state = make_state(np.zeros((20, 20), dtype=np.int64))

    first = step_cells(automaton, state, np.random.default_rng(42))
    second = step_cells(automaton, state, np.random.default_rng(42))
    np.testing.assert_array_equal(first["cells"], second["cells"])


def test_one_full_matrix_is_drawn_per_probabilistic_rule():
    # Two rules with probability < 1 and one with probability 1:
    # exactly two matrices must be drawn, even though the second rule matches no cell.
    automaton = make_automaton(
        [
            make_rule([0], 1, probability=0.3),
            make_rule([5], 1, probability=0.3),
            make_rule([1], 0, probability=1.0),
        ],
        MOORE,
        boundary="toroidal",
    )
    shape = (4, 6)
    rng_engine = np.random.default_rng(7)
    rng_manual = np.random.default_rng(7)

    step_cells(automaton, make_state(np.zeros(shape, dtype=np.int64)), rng_engine)
    rng_manual.random(shape)
    rng_manual.random(shape)

    assert rng_engine.random() == rng_manual.random()
